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
#pragma once

#include <optional>

#include <cstddef>

#include "Eigen/Dense"
#include "extensions/centerpose/components/centerpose_types.hpp"
#include "extensions/centerpose/components/cuboid3d.hpp"

namespace nvidia {
namespace isaac {
namespace centerpose {

struct PnPResult {
  struct Pose {
    Eigen::Vector3f position{};
    Eigen::Quaternionf orientation{};
  };

  Pose pose;

  Eigen::MatrixXfRM projected_points{};
  float reprojection_error{};
};

class CuboidPnPSolver {
 public:
  explicit CuboidPnPSolver(const Cuboid3d& cuboid3d)
      : scaling_factor_{1.0f},
        camera_matrix_{Eigen::Matrix3f::Zero()},
        cuboid3d_{cuboid3d},
        dist_coeffs_{0.0f, 0.0f, 0.0f, 0.0f},
        min_required_points_{4} {}

  CuboidPnPSolver(const float scaling_factor, const Cuboid3d& cuboid3d)
      : scaling_factor_{scaling_factor},
        camera_matrix_{Eigen::Matrix3f::Zero()},
        cuboid3d_{cuboid3d},
        dist_coeffs_{0.0f, 0.0f, 0.0f, 0.0f},
        min_required_points_{4} {}

  CuboidPnPSolver(
      const float scaling_factor, const Eigen::Matrix3f& camera_matrix, const Cuboid3d& cuboid3d)
      : scaling_factor_{scaling_factor},
        camera_matrix_{camera_matrix},
        cuboid3d_{cuboid3d},
        dist_coeffs_{0.0f, 0.0f, 0.0f, 0.0f},
        min_required_points_{4} {}

  CuboidPnPSolver(
      const float scaling_factor, const Eigen::Matrix3f& camera_matrix,
      const Eigen::Vector4f& dist_coeffs, const Cuboid3d& cuboid3d)
      : scaling_factor_{scaling_factor},
        camera_matrix_{camera_matrix},
        cuboid3d_{cuboid3d},
        dist_coeffs_{dist_coeffs},
        min_required_points_{4} {}

  CuboidPnPSolver(
      const float scaling_factor, const Eigen::Matrix3f& camera_matrix,
      const Eigen::Vector4f& dist_coeffs, const Cuboid3d& cuboid3d, size_t min_required_points)
      : scaling_factor_{scaling_factor},
        camera_matrix_{camera_matrix},
        cuboid3d_{cuboid3d},
        dist_coeffs_{dist_coeffs},
        min_required_points_{min_required_points} {}

  void setCameraMatrix(const Eigen::Matrix3f& camera_matrix) { camera_matrix_ = camera_matrix; }

  void setDistCoeffs(const Eigen::Vector4f& dist_coeffs) { dist_coeffs_ = dist_coeffs; }

  // 0 corresponds to SOLVEPNP_ITERATIVE
  std::optional<PnPResult> solvePnP(
      const Eigen::MatrixXfRM& cuboid2d_points, const int pnp_algorithm = 0);

 private:
  float scaling_factor_{};
  Eigen::Matrix3f camera_matrix_;
  Cuboid3d cuboid3d_;
  Eigen::Vector4f dist_coeffs_;
  size_t min_required_points_;
};

}  // namespace centerpose
}  // namespace isaac
}  // namespace nvidia
