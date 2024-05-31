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

#include <iostream>

#include <Eigen/Dense>

#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/Importer.hpp>

#include "cuda.h"
#include "cuda_runtime.h"

#include "gxf/cuda/cuda_stream.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"
#include "gxf/core/entity.hpp"
#include "gxf/core/expected.hpp"
#include "gxf/core/gxf.h"
#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace {
#define CHECK_CUDA_ERRORS(result) \
  { CheckCudaErrors(result, __FILE__, __LINE__); }
inline void CheckCudaErrors(cudaError_t result, const char* filename, int line_number) {
  if (result != cudaSuccess) {
    std::cout << "CUDA Error: " + std::string(cudaGetErrorString(result)) +
                     " (error code: " + std::to_string(result) + ") at " + std::string(filename) +
                     " in line " + std::to_string(line_number);
  }
}
}  // namespace

// Finds the minimum and maximum vertex from the mesh loaded by assimp
std::pair<Eigen::Vector3f, Eigen::Vector3f> FindMinMaxVertex(const aiMesh* mesh);

// Calculates the diameter of the mesh loaded by assimp
float CalcMeshDiameter(const aiMesh* mesh);

// Updates/adds timestamp entity to the ouput gxf message entity from the input gxf message entity
gxf::Expected<void> AddInputTimestampToOutput(gxf::Entity& output, gxf::Entity input);

}  // namespace nvidia

#endif  // NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_UTILS_HPP_