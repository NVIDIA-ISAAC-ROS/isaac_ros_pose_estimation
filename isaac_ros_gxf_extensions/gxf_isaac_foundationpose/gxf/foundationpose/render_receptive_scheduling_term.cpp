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

#include "render_receptive_scheduling_term.hpp"

namespace nvidia {
namespace isaac_ros {

gxf_result_t RenderReceptiveSchedulingTerm::registerInterface(
  nvidia::gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(render_impl_, "render_impl",
    "Render Implementation Handle");
  return gxf::ToResultCode(result);
}

gxf_result_t RenderReceptiveSchedulingTerm::check_abi(int64_t timestamp,
  gxf::SchedulingConditionType* type, int64_t* target_timestamp) const {
  auto maybe_is_accepting_request = render_impl_.get()->isAcceptingRequest();
  if (!maybe_is_accepting_request) {
    GXF_LOG_ERROR("Render isAcceptingRequest had unexpected return");
    return GXF_FAILURE;
  }
  const auto& is_accepting_request = maybe_is_accepting_request.value();
  *type = is_accepting_request ? gxf::SchedulingConditionType::READY :
    gxf::SchedulingConditionType::WAIT;
  return GXF_SUCCESS;
}

gxf_result_t RenderReceptiveSchedulingTerm::onExecute_abi(int64_t dt) {
  return GXF_SUCCESS;
}

}  // namespace isaac_ros
}  // namespace nvidia
