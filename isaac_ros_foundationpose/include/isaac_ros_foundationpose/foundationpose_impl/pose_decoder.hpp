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

#ifndef ISAAC_ROS_FOUNDATIONPOSE__FOUNDATIONPOSE_IMPL__POSE_DECODER_HPP_
#define ISAAC_ROS_FOUNDATIONPOSE__FOUNDATIONPOSE_IMPL__POSE_DECODER_HPP_

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "Eigen/Dense"

#include "isaac_ros_foundationpose/foundationpose_impl/mesh_loader.hpp"
#include "isaac_ros_foundationpose/foundationpose_impl/pose_transformer.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace foundationpose
{

struct DeviceScoreBatchView
{
  const float * data{nullptr};
  uint32_t count{0};
};

struct RosMessageStamp
{
  std::string frame_id;
  int32_t sec{0};
  uint32_t nanosec{0};
};

struct DecodeResult
{
  // Public output: pose of the ORIGINAL (uncentered) mesh in the camera frame.
  vision_msgs::msg::Detection3DArray detection3d_array;
  // Internal feedback: pose of the CENTERED mesh in the camera frame. Equivalent to
  // Python FoundationPose's `self.pose_last`. The tracker expects this form as its
  // initial pose because the refine/score networks operate on the centered mesh.
  Eigen::Matrix4f pose_matrix;
};

// Selects the best-scoring pose hypothesis and converts to Detection3D output.
// Extracted from FoundationposeDecoder GXF codelet.
class PoseDecoder
{
public:
  explicit PoseDecoder(cudaStream_t stream);
  ~PoseDecoder() = default;

  PoseDecoder(const PoseDecoder &) = delete;
  PoseDecoder & operator=(const PoseDecoder &) = delete;

  // Decode with scoring (detection mode): argmax over scores, pick best pose.
  DecodeResult decode(
    DevicePoseBatchView poses,
    DeviceScoreBatchView scores,
    std::shared_ptr<const MeshData> mesh_data,
    const RosMessageStamp & stamp);

  // Decode without scoring (tracking mode): single pose, no argmax.
  DecodeResult decodeTracking(
    DevicePoseBatchView poses,
    std::shared_ptr<const MeshData> mesh_data,
    const RosMessageStamp & stamp);

private:
  cudaStream_t stream_;
};

}  // namespace foundationpose
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_FOUNDATIONPOSE__FOUNDATIONPOSE_IMPL__POSE_DECODER_HPP_
