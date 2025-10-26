// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "extensions/centerpose/components/cuboid3d.hpp"

#include "extensions/centerpose/components/centerpose_types.hpp"

namespace nvidia {
namespace isaac {
namespace centerpose {

namespace {

std::array<Eigen::Vector3f, static_cast<size_t>(CuboidVertexType::TOTAL_CORNER_VERTEX_COUNT)>
GenerateVertices(const Eigen::Vector3f& center_location, const Eigen::Vector3f& size_3d) {
  const float width{size_3d[0]};
  const float height{size_3d[1]};
  const float depth{size_3d[2]};

  const float cx{center_location[0]};
  const float cy{center_location[1]};
  const float cz{center_location[2]};

  const float right{cx + width / 2.0f};
  const float left{cx - width / 2.0f};

  const float top{cy + height / 2.0f};
  const float bottom{cy - height / 2.0f};

  const float front{cz + depth / 2.0f};
  const float rear{cz - depth / 2.0f};

  return {Eigen::Vector3f{left, bottom, rear},  Eigen::Vector3f{left, bottom, front},
          Eigen::Vector3f{left, top, rear},     Eigen::Vector3f{left, top, front},

          Eigen::Vector3f{right, bottom, rear}, Eigen::Vector3f{right, bottom, front},
          Eigen::Vector3f{right, top, rear},    Eigen::Vector3f{right, top, front}};
}
}  // namespace

Cuboid3d::Cuboid3d()
    : center_location_{0.0f, 0.0f, 0.0f},
      size_3d_{1.0f, 1.0f, 1.0f},
      vertices_{GenerateVertices(center_location_, size_3d_)} {}

Cuboid3d::Cuboid3d(const Eigen::Vector3f& size_3d)
    : center_location_{0.0f, 0.0f, 0.0f},
      size_3d_{size_3d},
      vertices_{GenerateVertices(center_location_, size_3d)} {}

const std::array<Eigen::Vector3f, static_cast<size_t>(CuboidVertexType::TOTAL_CORNER_VERTEX_COUNT)>&
Cuboid3d::vertices() const {
  return vertices_;
}

std::array<Eigen::Vector3f, static_cast<size_t>(CuboidVertexType::TOTAL_CORNER_VERTEX_COUNT)>
Cuboid3d::vertices() {
  return vertices_;
}

}  // namespace centerpose
}  // namespace isaac
}  // namespace nvidia
