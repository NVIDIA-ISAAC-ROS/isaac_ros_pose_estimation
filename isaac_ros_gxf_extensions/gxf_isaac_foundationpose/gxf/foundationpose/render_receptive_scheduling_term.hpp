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

#ifndef NVIDIA_ISAAC_ROS_EXTENSIONS_RENDER_RECEPTIVE_SCHEDULING_TERM_HPP_
#define NVIDIA_ISAAC_ROS_EXTENSIONS_RENDER_RECEPTIVE_SCHEDULING_TERM_HPP_

#pragma once
#include <utility>
#include <vector>

#include "foundationpose_render.hpp"
#include "gxf/core/handle.hpp"
#include "gxf/std/scheduling_term.hpp"

namespace nvidia {
namespace isaac_ros {
/**
 * @brief Scheduling term which permits execution only when the Render Component is able to
 * accept new requests.
 *
 */
class RenderReceptiveSchedulingTerm : public nvidia::gxf::SchedulingTerm {
 public:
  /**
   * @brief Register Parameters.
   *
   * @param registrar
   * @return gxf_result_t
   */
  gxf_result_t registerInterface(nvidia::gxf::Registrar* registrar) override;
  /**
   * @brief Only when an render can accept new input, SchedulingCondition is
   * Ready
   *
   * @param[in] timestamp Unused
   * @param[out] type Ready (if render can accept new incoming input) or Wait (otherwise)
   * @param[out] target_timestamp Unmodified
   * @return gxf_result_t
   */
  gxf_result_t check_abi(int64_t timestamp, nvidia::gxf::SchedulingConditionType* type,
    int64_t* target_timestamp) const override;
  /**
   * @brief Returns success
   *
   * @param dt Unused
   * @return gxf_result_t
   */
  gxf_result_t onExecute_abi(int64_t dt) override;

 private:
  nvidia::gxf::Parameter<nvidia::gxf::Handle<nvidia::isaac_ros::FoundationposeRender>> render_impl_;
};

}  // namespace isaac_ros
}  // namespace nvidia

#endif  // NVIDIA_ISAAC_ROS_EXTENSIONS_RENDER_RECEPTIVE_SCHEDULING_TERM_HPP_