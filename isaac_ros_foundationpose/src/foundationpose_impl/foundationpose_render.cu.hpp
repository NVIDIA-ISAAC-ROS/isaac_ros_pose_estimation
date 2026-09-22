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

#ifndef NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_RENDER_CUDA_HPP_
#define NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_RENDER_CUDA_HPP_

#include <cstdint>
#include <limits>

#include "cuda.h"
#include "cuda_runtime.h"
#include "Eigen/Dense"
#include "isaac_ros_foundationpose/foundationpose_impl/pose_sampler.hpp"
#include "isaac_ros_common/cuda_stream.hpp"

#include "nvdiffrast/common/common.h"
#include "nvdiffrast/common/cudaraster/CudaRaster.hpp"
#include "nvdiffrast/common/interpolate.h"
#include "nvdiffrast/common/rasterize.h"
#include "nvdiffrast/common/texture.h"

namespace nvidia
{
namespace isaac_ros
{

void clamp(cudaStream_t stream, float * input, float min_value, float max_value, int N);
void threshold_and_downscale_pointcloud(
  cudaStream_t stream, float * pointcloud_input, float * pose_array_input, int N, int n_points,
  float downscale_factor,
  float min_depth, float max_depth);
void compute_vertex_diffuse(
  cudaStream_t stream, float * diffuse_output, const float * normal_input,
  const float * pose_array_input, int N, int num_vertices);
void apply_diffuse_lighting(
  cudaStream_t stream, float * color_input, const float * diffuse_input, int N, int n_points);
// Count how many points in pc_device (n_points * 3 floats, XYZ) lie within radius r of (cx, cy, cz).
// Result is written to count_device (int on GPU). Caller must zero count_device before calling if reusing.
void count_points_within_radius(
  cudaStream_t stream, foundationpose::DeviceImageView<float> point_cloud,
  const Eigen::Vector3f & center, float radius_sq, int * count_device);
void transform_pts(
  cudaStream_t stream, float * output, const float * pts, const float * tfs, int pts_num,
  int pts_channel, int tfs_num, int tfs_dim);
void generate_pose_clip(
  cudaStream_t stream, float * d_pose_clip, const float * d_pose, const float * bbox2d,
  const float * d_mesh_vertices,
  const Eigen::Matrix4f & projection_mat, int rgb_H, int rgb_W, int n_pts, int n_poses);
void concat(
  cudaStream_t stream, float * input_a, float * input_b, float * output, int N, int H,
  int W, int C1, int C2);

void rasterize(
  cudaStream_t stream, CR::CudaRaster * cr, float * pos_ptr, int32_t * tri_ptr, float * out,
  int pos_count, int tri_count,
  int H, int W, int C);
void interpolate(
  cudaStream_t stream, float * attr_ptr, float * rast_ptr, int32_t * tri_ptr, float * out,
  int num_vertices,
  int num_triangles, int attr_dim, int H, int W, int C, int attr_bc = 0);
void texture(
  cudaStream_t stream, float * tex_ptr, float * uv_ptr, float * out, int tex_height, int tex_width,
  int tex_channel,
  int tex_depth, int H, int W, int N);

}  // namespace isaac_ros
}  // namespace nvidia
#endif  // NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_RENDER_CUDA_HPP_
