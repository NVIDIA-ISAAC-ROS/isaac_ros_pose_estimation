/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_SYNCHRONIZATION_HPP_
#define NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_SYNCHRONIZATION_HPP_

#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "gxf/core/gxf.h"

#include "common/assert.hpp"
#include "common/type_name.hpp"
#include "gxf/core/component.hpp"
#include "gxf/core/handle.hpp"
#include "gxf/core/parameter_parser_std.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace isaac_ros {

// A codelet to synchronize incoming messages from multiple receivers and
// publish them on to multiple transmitters. The number of receivers and transmitters
// must be equal in number and incoming messages must have a valid nvidia::gxf::Timestamp
// component in them. Only one message per transmitter is published per tick of the codelet.
// It also has the functionality to cache camera info and pipe through the same message
// with every image frame so that there is no sync happening. It also has the functionality to
// only send in the latest synced pair of messages at a fixed hz. This is useful since Foundation
// Pose cannot handle input arriving faster than 1 hz and still retain realtime rates on slower
// devices like the AGX Orin and older GPUs.

class FoundationPoseSynchronization : public gxf::Codelet {
 public:
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t start() override;
  gxf_result_t tick() override;

  gxf_result_t dropMessagesIfQueueIsFull();

  gxf_result_t useCachedCameraInfo();

  gxf_result_t discardOldMessages();

 private:
  gxf::Parameter<std::vector<gxf::Handle<gxf::Receiver>>> inputs_;
  gxf::Parameter<std::vector<gxf::Handle<gxf::Transmitter>>> outputs_;
  gxf::Parameter<gxf::Handle<gxf::Receiver>> pose_estimate_rx_;
  gxf::Parameter<int64_t> sync_threshold_;
  gxf::Parameter<int64_t> sync_policy_;
  gxf::Parameter<bool> discard_old_messages_;
  gxf::Parameter<int64_t> discard_time_ms_;
  gxf::Parameter<int64_t> pose_estimation_timeout_ms_;

  std::unordered_map<size_t, gxf::Entity> message_cache_;

  int64_t start_time_;
  bool send_new_message_;
};

}  // namespace  gxf
}  // namespace nvidia

#endif
