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

#ifndef ISAAC_ROS_FOUNDATIONPOSE__FOUNDATIONPOSE_IMPL__POSE_TRANSFORMER_HPP_
#define ISAAC_ROS_FOUNDATIONPOSE__FOUNDATIONPOSE_IMPL__POSE_TRANSFORMER_HPP_

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "isaac_ros_foundationpose/foundationpose_impl/mesh_loader.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace foundationpose
{

struct DevicePoseBatchView
{
  float * data{nullptr};
  uint32_t count{0};

  static constexpr uint32_t kFloatsPerPose{16};

  float * pose(uint32_t index) const
  {
    return data + static_cast<size_t>(index) * kFloatsPerPose;
  }
};

struct RefineDeltaBatchView
{
  const float * translation{nullptr};
  const float * rotation{nullptr};
  uint32_t pose_count{0};
};

// Applies the refine network's SE(3) delta predictions to pose hypotheses.
// Extracted from FoundationposeTransformation GXF codelet.
class PoseTransformer
{
public:
  explicit PoseTransformer(float rot_normalizer, cudaStream_t stream);
  ~PoseTransformer() = default;

  PoseTransformer(const PoseTransformer &) = delete;
  PoseTransformer & operator=(const PoseTransformer &) = delete;

  // Apply refine deltas to a batch of poses in-place on GPU. No host round-trip.
  void applyDeltas(
    DevicePoseBatchView poses,
    RefineDeltaBatchView deltas,
    MeshGpuView mesh);

private:
  float rot_normalizer_;
  cudaStream_t stream_;
};

}  // namespace foundationpose
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_FOUNDATIONPOSE__FOUNDATIONPOSE_IMPL__POSE_TRANSFORMER_HPP_
