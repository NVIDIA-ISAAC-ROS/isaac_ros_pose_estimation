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

#ifndef ISAAC_ROS_FOUNDATIONPOSE__FOUNDATIONPOSE_IMPL__POSE_RENDERER_HPP_
#define ISAAC_ROS_FOUNDATIONPOSE__FOUNDATIONPOSE_IMPL__POSE_RENDERER_HPP_

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>

#include "Eigen/Dense"

#include "isaac_ros_foundationpose/foundationpose_impl/mesh_loader.hpp"
#include "isaac_ros_foundationpose/foundationpose_impl/pose_sampler.hpp"
#include "isaac_ros_foundationpose/foundationpose_impl/pose_transformer.hpp"

// Forward declare CudaRaster from nvdiffrast
namespace CR {class CudaRaster;}

namespace nvidia
{
namespace isaac_ros
{
namespace foundationpose
{

struct PoseRendererParams
{
  float crop_ratio{1.2f};
  float min_depth{0.1f};
  float max_depth{2.0f};
  uint32_t resized_height{160};
  uint32_t resized_width{160};
};

struct RefineRenderOutputView
{
  float * rendered{nullptr};
  float * observed{nullptr};
  uint32_t pose_count{0};
  ImageDim render_size;
};

// Renders synthetic mesh views and crops observed images for DNN comparison.
// Extracted from FoundationposeRender GXF codelet.
class PoseRenderer
{
public:
  PoseRenderer(const PoseRendererParams & params, cudaStream_t stream);
  ~PoseRenderer();

  PoseRenderer(const PoseRenderer &) = delete;
  PoseRenderer & operator=(const PoseRenderer &) = delete;

  // Render N poses into caller-provided output buffers.
  // Each output buffer must be at least N * H * W * 2 * kNumChannels floats.
  // The two output tensors have layout [N, H, W, 2*kNumChannels=6]:
  // 6 channels = rendered RGB(3) + rendered XYZ(3) for `rendered_out`;
  //              observed RGB(3) + observed XYZ(3) for `observed_out`.
  // All GPU writes are queued on the renderer's stream; caller is expected to
  // hold the corresponding tensor buffer WriteHandles open across this call so
  // their dtor records a completion event AFTER the renderer's queued kernels.
  void renderRefine(
    DevicePoseBatchView poses,
    const FrameObservationView & frame,
    MeshGpuView mesh,
    RefineRenderOutputView output);

  // Number of float elements per pose in each output tensor (= H * W * 6).
  size_t floatsPerPose() const
  {
    return static_cast<size_t>(params_.resized_height) * params_.resized_width *
           2 * kNumChannels;
  }

  static constexpr size_t kNumChannels = 3;

private:
  void allocateDeviceMemory(
    uint32_t N, uint32_t H, uint32_t W, uint32_t C, uint32_t num_vertices);
  void freeDeviceMemory();

  PoseRendererParams params_;
  cudaStream_t stream_;
  CR::CudaRaster * cr_{nullptr};

  // GPU scratch buffers reused across renderRefine() calls.
  float * pose_device_{nullptr};
  float * pose_clip_device_{nullptr};
  float * pts_cam_device_{nullptr};
  float * rast_out_device_{nullptr};
  float * texcoords_out_device_{nullptr};
  float * diffuse_vertex_device_{nullptr};
  float * diffuse_map_device_{nullptr};
  float * color_device_{nullptr};
  float * xyz_map_device_{nullptr};
  float * transformed_rgb_device_{nullptr};
  float * transformed_xyz_map_device_{nullptr};
  uint8_t * wp_image_device_{nullptr};
  float * trans_matrix_device_{nullptr};
  float * bbox2d_device_{nullptr};

  bool device_mem_cached_{false};
  size_t num_vertices_cache_{0};
  uint32_t batch_size_cache_{0};

  // Normalized texture cache (reused when mesh/size unchanged)
  float * norm_tex_device_{nullptr};
  size_t norm_tex_num_verts_{0};
};

}  // namespace foundationpose
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_FOUNDATIONPOSE__FOUNDATIONPOSE_IMPL__POSE_RENDERER_HPP_
