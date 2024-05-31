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
#pragma once

#ifndef NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_DECODER_CUDA_HPP_
#define NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_DECODER_CUDA_HPP_

#include <cstdint>
#include <limits>

#include "cuda.h"
#include "cuda_runtime.h"

namespace nvidia {
namespace isaac_ros {

int getMaxScoreIndex(cudaStream_t cuda_stream, float* scores, int N);

}  // namespace isaac_ros
}  // namespace nvidia

#endif  // NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_DECODER_CUDA_HPP_