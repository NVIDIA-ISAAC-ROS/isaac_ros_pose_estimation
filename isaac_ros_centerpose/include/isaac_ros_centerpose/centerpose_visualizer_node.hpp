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

#ifndef ISAAC_ROS_CENTERPOSE__CENTERPOSE_VISUALIZER_NODE_HPP_
#define ISAAC_ROS_CENTERPOSE__CENTERPOSE_VISUALIZER_NODE_HPP_

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

class CenterPoseVisualizerNode : public nitros::NitrosNode
{
public:
  explicit CenterPoseVisualizerNode(rclcpp::NodeOptions options = rclcpp::NodeOptions());
  ~CenterPoseVisualizerNode();

  void postLoadGraphCallback() override;

private:
  // Whether to draw the axes onto the pose or not
  bool show_axes_;

  // The bounding box color
  int64_t bounding_box_color_;
};

}  // namespace centerpose
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_CENTERPOSE__CENTERPOSE_VISUALIZER_NODE_HPP_
