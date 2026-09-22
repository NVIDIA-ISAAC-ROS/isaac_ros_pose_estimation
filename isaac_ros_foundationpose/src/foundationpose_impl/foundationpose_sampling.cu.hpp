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

#ifndef NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_SAMPLING_CUDA_HPP_
#define NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_SAMPLING_CUDA_HPP_

#include <cstdint>
#include <limits>

#include "cuda.h"
#include "cuda_runtime.h"
#include "isaac_ros_foundationpose/foundationpose_impl/pose_sampler.hpp"

namespace nvidia
{
namespace isaac_ros
{

void erode_depth(
  cudaStream_t stream, foundationpose::DeviceImageView<const float> depth,
  float * out, int radius = 2,
  float depth_diff_thres = 0.001,
  float ratio_thres = 0.8, float zfar = 100);
void bilateral_filter_depth(
  cudaStream_t stream, foundationpose::DeviceImageView<const float> depth,
  float * out, float zfar = 100, int radius = 2,
  float sigmaD = 2,
  float sigmaR = 100000);

// Generate an interleaved (H, W, 3) xyz map on GPU directly from a depth image (H, W) and
// camera intrinsics K (row-major 3x3). Pixels with depth <= 0 produce (0, 0, 0).
void depth_to_xyz_map(
  cudaStream_t stream, foundationpose::DeviceImageView<const float> depth,
  foundationpose::DeviceImageView<float> xyz_map,
  foundationpose::CameraIntrinsics intrinsics);

// GPU implementation of guessTranslation. Inputs: depth (H, W) and mask (H, W, uint8).
// Outputs 4 floats into center_and_flag_device: [cx, cy, cz, flag] where flag > 0 means valid.
// depth_scratch must hold at least H*W floats (used to stage masked depths for median via
// partial reduction).
void guess_translation_gpu(
  cudaStream_t stream, foundationpose::DeviceImageView<const float> depth,
  foundationpose::DeviceImageView<const uint8_t> mask,
  foundationpose::CameraIntrinsics intrinsics,
  float min_depth,
  float * depth_scratch,
  float * center_and_flag_device);

// Applies refine deltas to poses on-device. poses_device is [N, 4, 4] row-major (column-major
// Eigen stores (i, j) at [i + j*4], so 'position' is at offsets 12,13,14 like the renderer uses).
// trans_delta is [N, 3], rot_delta is [N, 3]. mesh_diameter and rot_normalizer are the same
// scalars as the CPU version. Writes result back into poses_device in place.
void apply_deltas_gpu(
  cudaStream_t stream, float * poses_device, const float * trans_delta, const float * rot_delta,
  int num_poses, float mesh_diameter, float rot_normalizer);

}  // namespace isaac_ros
}  // namespace nvidia

#endif  // NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_SAMPLING_CUDA_HPP_
