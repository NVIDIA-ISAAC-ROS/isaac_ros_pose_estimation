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

#ifndef NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_UTILS_HPP_
#define NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_UTILS_HPP_

#pragma once

// C++ system headers
#include <memory>
#include <string>
#include <utility>

// External dependencies
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

// GXF headers
#include "gxf/core/entity.hpp"
#include "gxf/core/expected.hpp"
#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace isaac_ros {

#define CHECK_CUDA_ERRORS(result) \
  { CheckCudaErrors(result, __FILE__, __LINE__); }
inline void CheckCudaErrors(cudaError_t result, const char* filename, int line_number) {
  if (result != cudaSuccess) {
    std::cout << "CUDA Error: " + std::string(cudaGetErrorString(result)) +
                     " (error code: " + std::to_string(result) + ") at " + std::string(filename) +
                     " in line " + std::to_string(line_number);
  }
}


inline gxf::Expected<void> AddInputTimestampToOutput(gxf::Entity& output, gxf::Entity input) {
  std::string named_timestamp{"timestamp"};
  std::string unnamed_timestamp{""};
  auto maybe_input_timestamp = input.get<gxf::Timestamp>(named_timestamp.c_str());

  // Try to get a named timestamp from the input entity
  if (!maybe_input_timestamp) {
    maybe_input_timestamp = input.get<gxf::Timestamp>(unnamed_timestamp.c_str());
  }
  // If there is no named timestamp, try to get a unnamed timestamp from the input entity
  if (!maybe_input_timestamp) {
    GXF_LOG_ERROR("Failed to get input timestamp!");
    return gxf::ForwardError(maybe_input_timestamp);
  }

  // Try to get a named timestamp from the output entity
  auto maybe_output_timestamp = output.get<gxf::Timestamp>(named_timestamp.c_str());
  // If there is no named timestamp, try to get a unnamed timestamp from the output entity
  if (!maybe_output_timestamp) {
    maybe_output_timestamp = output.get<gxf::Timestamp>(unnamed_timestamp.c_str());
  }

  // If there is no unnamed timestamp also, then add a named timestamp to the output entity
  if (!maybe_output_timestamp) {
    maybe_output_timestamp = output.add<gxf::Timestamp>(named_timestamp.c_str());
    if (!maybe_output_timestamp) {
      GXF_LOG_ERROR("Failed to add timestamp to output message!");
      return gxf::ForwardError(maybe_output_timestamp);
    }
  }

  *maybe_output_timestamp.value() = *maybe_input_timestamp.value();
  return gxf::Success;
}

}  // namespace isaac_ros
}  // namespace nvidia

#endif  // NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_UTILS_HPP_