// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace nvidia {
namespace isaac_ros {

void erode_depth(
    cudaStream_t stream, float* depth, float* out, int H, int W, int radius = 2, float depth_diff_thres = 0.001,
    float ratio_thres = 0.8, float zfar = 100);
void bilateral_filter_depth(
    cudaStream_t stream, float* depth, float* out, int H, int W, float zfar = 100, int radius = 2, float sigmaD = 2,
    float sigmaR = 100000);

}  // namespace isaac_ros
}  // namespace nvidia

#endif  // NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_SAMPLING_CUDA_HPP_