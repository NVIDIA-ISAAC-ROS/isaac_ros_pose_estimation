// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef ISAAC_ROS_FOUNDATIONPOSE__FOUNDATIONPOSE_IMPL__POSE_SAMPLER_HPP_
#define ISAAC_ROS_FOUNDATIONPOSE__FOUNDATIONPOSE_IMPL__POSE_SAMPLER_HPP_

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "Eigen/Dense"

#include "isaac_ros_foundationpose/foundationpose_impl/mesh_loader.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace foundationpose
{

struct ImageDim
{
  uint32_t height{0};
  uint32_t width{0};

  size_t numPixels() const
  {
    return static_cast<size_t>(height) * width;
  }

  bool operator==(const ImageDim & other) const
  {
    return height == other.height && width == other.width;
  }
};

template<typename T>
struct DeviceImageView
{
  T * data{nullptr};
  ImageDim size;
  size_t row_stride_bytes{0};
  uint32_t channels{1};

  size_t minimumRowBytes() const
  {
    return static_cast<size_t>(size.width) * channels * sizeof(T);
  }
};

struct CameraIntrinsics
{
  float fx{0.0f};
  float fy{0.0f};
  float cx{0.0f};
  float cy{0.0f};

  Eigen::Matrix3f matrix() const
  {
    Eigen::Matrix3f result;
    result << fx, 0.0f, cx, 0.0f, fy, cy, 0.0f, 0.0f, 1.0f;
    return result;
  }
};

struct FrameObservationView
{
  DeviceImageView<const uint8_t> rgb;
  DeviceImageView<const float> depth;
  DeviceImageView<const uint8_t> mask;
  DeviceImageView<float> point_cloud;
  CameraIntrinsics intrinsics;
};

struct PoseSamplerParams
{
  uint32_t max_hypothesis{252};
  float min_depth{0.1f};
  std::vector<std::string> symmetry_axes;
  std::vector<std::string> symmetry_planes;  // deprecated: converted to symmetry_axes
  std::vector<std::string> fixed_axis_angles;
  std::vector<std::string> fixed_translations;
};

struct SamplingResult
{
  std::vector<float> poses;   // [N * 4 * 4] column-major flattened
  int32_t total_poses{0};
  int32_t batch_size{0};      // poses per batch (total / kNumBatches)
  int32_t num_batches{0};
};

// Generates initial 6-DoF pose hypotheses from depth, mask, and point cloud.
// Extracted from FoundationposeSampling GXF codelet.
class PoseSampler
{
public:
  PoseSampler(const PoseSamplerParams & params, cudaStream_t stream);
  ~PoseSampler();

  PoseSampler(const PoseSampler &) = delete;
  PoseSampler & operator=(const PoseSampler &) = delete;

  // Generate pose hypotheses from depth + mask + point cloud + intrinsics.
  // Returns flattened 4x4 pose matrices and batch metadata.
  SamplingResult sample(
    const FrameObservationView & frame,
    std::shared_ptr<const MeshData> mesh_data);

  void updateParams(const PoseSamplerParams & params) {params_ = params;}

private:
  PoseSamplerParams params_;
  cudaStream_t stream_;

  float * erode_depth_device_{nullptr};
  float * bilateral_filter_depth_device_{nullptr};
  float * center_flag_device_{nullptr};      // [cx, cy, cz, flag] on GPU
  float * center_flag_host_pinned_{nullptr};  // pinned mirror for single 16B D2H
  bool device_mem_cached_{false};
  ImageDim cached_size_;

  static constexpr int kNumBatches = 6;
};

}  // namespace foundationpose
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_FOUNDATIONPOSE__FOUNDATIONPOSE_IMPL__POSE_SAMPLER_HPP_
