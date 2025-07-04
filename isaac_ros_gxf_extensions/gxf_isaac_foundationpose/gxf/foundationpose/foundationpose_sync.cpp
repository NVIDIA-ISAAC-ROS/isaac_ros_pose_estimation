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

#include "foundationpose_sync.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <vector>

#include "gxf/core/expected_macro.hpp"
#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace isaac_ros {

namespace {
constexpr const uint64_t kNanosecondsToMilliseconds = 1000000;
}

gxf_result_t FoundationPoseSynchronization::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(
      inputs_, "inputs", "Inputs",
      "All the inputs for synchronization, number of inputs must match that of the outputs.");
  result &= registrar->parameter(
      outputs_, "outputs", "Outputs",
      "All the outputs for synchronization, number of outputs must match that of the inputs.");
  result &= registrar->parameter(
      pose_estimate_rx_, "pose_estimate_rx", "Pose estimate rx",
      "Gets pose estimate result of last pose, we only send in data once we get this result");
  result &= registrar->parameter(
      sync_threshold_, "sync_threshold", "Synchronization threshold (ns)",
      "Synchronization threshold in nanoseconds. "
      "Messages will not be synchronized if timestamp difference is above the threshold. "
      "By default, timestamps should be identical for synchronization (default threshold = 0). "
      "Synchronization threshold will only work if maximum timestamp variation is much less "
      "than minimal delta between timestamps of subsequent messages in any input.",
      static_cast<int64_t>(0));
  result &= registrar->parameter(
      sync_policy_, "sync_policy", "Synchronization policy",
      "Synchronization policy, 0: choose oldest message from fastest moving queue"
      "1: choose newest message from slowest moving queue. 0 is good for processing most of the"
      "data at the cost of being potentially slower than realtime. 1 is good for finding sync"
      "points close to real time but at the cost of dropping more data.",
      static_cast<int64_t>(0));
  result &= registrar->parameter(
    discard_old_messages_, "discard_old_messages", "Discard old messages", "Discard old messages",
    false
  );
  result &= registrar->parameter(
    discard_time_ms_, "discard_time_ms", "Discard message if older by this time",
    "Discard old messages if older than this number represented in milliseconds",
    static_cast<int64_t>(1000) // 1 second
  );
  result &= registrar->parameter(
    pose_estimation_timeout_ms_, "pose_estimation_timeout_ms", "Pose estimation timeout",
    "This stores the timeout in milliseconds for pose estimation, such that the node sends"
    "data into foundation pose after X milliseconds", static_cast<int64_t>(5000) // 5 second
  );
  return gxf::ToResultCode(result);
}

int64_t getCurrentTimeInNanoSeconds() {
  const std::chrono::time_point<std::chrono::system_clock> now =
      std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  auto timestamp_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(duration);
  return timestamp_ns.count();
}

gxf_result_t FoundationPoseSynchronization::dropMessagesIfQueueIsFull() {
  // Clear all queues
  int  i = 0;
  for (const auto& rx : inputs_.get()) {
    RETURN_IF_ERROR(rx->sync());
    GXF_LOG_DEBUG("Input %ld | Size: %ld, will drop if queue is full", i, rx->size());
    i += 1;
    if (rx->size() == rx->capacity()) {
      RETURN_IF_ERROR(rx->receive());
    }
  }
  return GXF_SUCCESS;
}

gxf_result_t FoundationPoseSynchronization::start() {

  start_time_ = getCurrentTimeInNanoSeconds();
  if (inputs_.get().size() != outputs_.get().size()) {
    GXF_LOG_ERROR("Number of inputs for synchronization must match the number of outputs");
    return GXF_FAILURE;
  }
  if (inputs_.get().size() <= 1) {
    GXF_LOG_ERROR("Number of inputs/outputs should be more than 1");
    return GXF_FAILURE;
  }

  if (sync_policy_.get() != 0 && sync_policy_.get() != 1) {
    GXF_LOG_ERROR("Synchronization policy must be 0 or 1");
    return GXF_FAILURE;
  }
  send_new_message_ = true;
  return GXF_SUCCESS;
}

gxf_result_t FoundationPoseSynchronization::discardOldMessages() {
  for (const auto& rx : inputs_.get()) {
    RETURN_IF_ERROR(rx->sync());
    for (int i =0; i < rx->size(); i++) {
      auto message = rx->peek(i);
      if (!message) {
        GXF_LOG_ERROR("Could not get message");
        return GXF_ENTITY_COMPONENT_NOT_FOUND;
      }
      auto timestamp_components = message->findAllHeap<gxf::Timestamp>();
      if (!timestamp_components) {
        return gxf::ToResultCode(timestamp_components);
      }
      if (0 == timestamp_components->size()) {
        GXF_LOG_ERROR("No timestamp found from the input message");
        return GXF_ENTITY_COMPONENT_NOT_FOUND;
      }

      int64_t timestamp = timestamp_components->front().value()->acqtime;
      int64_t current_time = getCurrentTimeInNanoSeconds();

      int64_t delay_ns = (current_time - timestamp); // how far back in time from current
      int64_t delay_ms = delay_ns / kNanosecondsToMilliseconds;
      GXF_LOG_DEBUG("Timestamp (how many ms old): %ld", delay_ms);
      if (delay_ms > discard_time_ms_.get()) {
        GXF_LOG_DEBUG("Timestamp REMOVED (how many ms old): %ld", delay_ms);
        RETURN_IF_ERROR(rx->receive());
      } else {
        break;
      }
    }
  }
  return GXF_SUCCESS;
}

gxf_result_t FoundationPoseSynchronization::tick() {

  bool pose_estimate_received = pose_estimate_rx_.get()->size() > 0;
  // Clear cached data
  message_cache_.clear();

  if (pose_estimate_received) {
    while(pose_estimate_rx_.get()->size() > 0) {
      RETURN_IF_ERROR(pose_estimate_rx_.get()->receive());
    }
    int64_t delay_in_ms = 
      (getCurrentTimeInNanoSeconds() - start_time_) / kNanosecondsToMilliseconds;
    GXF_LOG_DEBUG(
      "Found pose estimate, ready to publish again, Foundation Pose last inference time: %ld ms",
      delay_in_ms);
    send_new_message_ = true;
  } else if ((getCurrentTimeInNanoSeconds() - start_time_) /
                 kNanosecondsToMilliseconds >
             pose_estimation_timeout_ms_.get()) {
    GXF_LOG_DEBUG("No pose estimate in 5 seconds, ready to publish again."
                  "Foundation Pose sampling might have not sent any data into inference");
    send_new_message_ = true;
  }

  if (discard_old_messages_.get()) {
    RETURN_IF_ERROR(discardOldMessages());
  }

  /* Check if all input queues have messages */
  bool all_messages_available = true;
  for (const auto& rx : inputs_.get()) {
    RETURN_IF_ERROR(rx->sync());
    if (rx->size() == 0) {
      /* Not all the inputs have messages, unexpected. */
      GXF_LOG_DEBUG("Not all the inputs have messages for synchronization!");
      all_messages_available = false;
      break;
    }
  }

  if (!all_messages_available) {
    return dropMessagesIfQueueIsFull();
  }

  /* First try reading the timestamps of available messages from all the inputs */
  std::vector<std::vector<int64_t>> acq_times;
  for (const auto& rx : inputs_.get()) {
    std::vector<int64_t> ack_times_per_receiver;
    for (size_t index = rx->size(); index > 0; index--) {
      auto msg_result = rx->peek(index - 1);
      if (msg_result) {
        auto timestamp_components = msg_result->findAllHeap<gxf::Timestamp>();
        if (!timestamp_components) {
          return gxf::ToResultCode(timestamp_components);
        }
        if (0 == timestamp_components->size()) {
          GXF_LOG_ERROR("No timestamp found from the input message");
          return GXF_ENTITY_COMPONENT_NOT_FOUND;
        }
        ack_times_per_receiver.push_back(timestamp_components->front().value()->acqtime);
      }
    }
    acq_times.push_back(ack_times_per_receiver);
  }

  /* find timestamp we're going to pick from the slowest moving queue*/
  int64_t candidate_sync_point = std::numeric_limits<int64_t>::max();
  if (sync_policy_.get() == 1) {
    /* find oldest timestamp from slowest moving queue, this makes sure with the messages we have
     we will find a sync point closest to real time, and not the potential non optimal time
     chosen by oldest message in fastest moving queue
    */
    for (const auto& timestamps : acq_times) {
      // Each timestamps.front() is the newest message from that queue
      // We pick the minimum of these newest timestamps
      if (!timestamps.empty() && timestamps.front() < candidate_sync_point) {
        candidate_sync_point = timestamps.front();
      }
    }
  } else if (sync_policy_.get() == 0) {
    /* find latest timestamp we're going to pick from the fastest moving queue*/
    candidate_sync_point = 0;
    for (const auto& timestamps : acq_times) {
      if (timestamps.back() > candidate_sync_point) { candidate_sync_point = timestamps.back(); }
    }
  }
  GXF_LOG_DEBUG("Candidate timestamp for syncing: %zd", candidate_sync_point);
  /* check if in all the monitored receivers there are messages that are within the threshold */
  uint32_t synchronized = 0;
  auto threshold = sync_threshold_.get();
  auto lower = candidate_sync_point;
  auto upper = candidate_sync_point;
  for (const auto& timestamps : acq_times) {
    auto it = std::find_if(timestamps.begin(), timestamps.end(),
                           [lower, upper, threshold](int64_t acq_time) {
                             return lower + threshold >= acq_time && acq_time >= upper - threshold;
                           });
    if (it != timestamps.end()) {
      synchronized++;
      lower = std::min(lower, *it);
      upper = std::max(upper, *it);
    }
  }
  const bool can_send = synchronized == inputs_.get().size();

  // poll the message queues based on the gathered timestamp information
  // the assumption here is the timestamps are in order
  bool found_new_match = false;
  for (size_t i = 0; i < acq_times.size(); i++) {
    auto rx = inputs_.get()[i];
    auto tx = outputs_.get()[i];
    const auto& timestamps = acq_times[i];
    // logging
    std::stringstream ss;
    for (auto ts : timestamps) ss << ts << ",";
    for (size_t j = timestamps.size(); j > 0; j--) {
      auto acq_time = timestamps[j - 1];
      if (acq_time < candidate_sync_point - threshold) {
        // drop the stale message
        rx->receive();
      } else if (can_send && acq_time <= candidate_sync_point + threshold) {
        // push the synchronized message
        auto message = rx->receive();
        if (!message) {
          GXF_LOG_ERROR("Receiver queue corrupted, message not found");
          return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE;
        }
        message_cache_[i] = std::move(message.value());
        found_new_match = true;
        break;
      }
    }
  }

  // Only send messages if we have received go ahead from downstream node and a match was found
  // Also this code will only run if wait_for_pose estimate is True
  if (send_new_message_ && found_new_match) {
    int64_t current_time = getCurrentTimeInNanoSeconds();
    // If time has been 1 second, then push it through
    for (size_t i = 0; i < acq_times.size(); i++) {
      auto message = std::move(message_cache_[i]);
      auto tx = outputs_.get()[i];
      tx->publish(message);
      // This might be wrong if its in sim
      int64_t delay_ms = (current_time - candidate_sync_point ) /kNanosecondsToMilliseconds;
      GXF_LOG_DEBUG(
        "Chosen sync timestamp is %ld ms behind current time (wrong if using sim time or rosbag)",
        delay_ms);
    }
    start_time_ = current_time;
    send_new_message_ = false;
  }
  message_cache_.clear();
  return GXF_SUCCESS;
}

}  // namespace gxf
}  // namespace nvidia
