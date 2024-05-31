// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_ROS_CENTERPOSE__CENTERPOSE_DECODER_NODE_HPP_
#define ISAAC_ROS_CENTERPOSE__CENTERPOSE_DECODER_NODE_HPP_

#include <string>
#include <vector>

#include "isaac_ros_nitros/nitros_node.hpp"
#include "rclcpp/rclcpp.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace centerpose
{

/**
 * @class CenterPoseDecoderNode
 * @brief This node performs pose estimation of a known category from a single RGB image
 *
 */
class CenterPoseDecoderNode : public nitros::NitrosNode
{
public:
  explicit CenterPoseDecoderNode(rclcpp::NodeOptions options = rclcpp::NodeOptions());
  ~CenterPoseDecoderNode();

  void postLoadGraphCallback() override;

private:
  // 2D keypoint decoding size. Width and then height.
  std::vector<int64_t> output_field_size_;

  // Scaling factor for cuboid
  double cuboid_scaling_factor_;

  // Score threshold
  double score_threshold_;

  // Object / instance name that is detected
  std::string object_name_;
};

}  // namespace centerpose
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_CENTERPOSE__CENTERPOSE_DECODER_NODE_HPP_
