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

#include "isaac_ros_foundationpose/foundationpose_impl/pose_transformer.hpp"

#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>

#include "foundationpose_sampling.cu.hpp"
#include "isaac_ros_common/cuda_stream.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace foundationpose
{

PoseTransformer::PoseTransformer(float rot_normalizer, cudaStream_t stream)
: rot_normalizer_(rot_normalizer), stream_(stream)
{
}

void PoseTransformer::applyDeltas(
  DevicePoseBatchView poses,
  RefineDeltaBatchView deltas,
  MeshGpuView mesh)
{
  if (poses.count != deltas.pose_count) {
    throw std::invalid_argument("Pose and delta batch sizes do not match");
  }
  nvidia::isaac_ros::apply_deltas_gpu(
    stream_, poses.data, deltas.translation, deltas.rotation,
    static_cast<int>(poses.count),
    mesh.diameter,
    rot_normalizer_);
  CHECK_CUDA_ERROR(cudaGetLastError(), "apply_deltas_gpu");
}

}  // namespace foundationpose
}  // namespace isaac_ros
}  // namespace nvidia
