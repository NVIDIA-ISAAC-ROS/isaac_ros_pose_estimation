// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_ROS_DOPE__DOPE_DECODER_NODE_HPP_
#define ISAAC_ROS_DOPE__DOPE_DECODER_NODE_HPP_

#include <string>
#include <vector>

#include "geometry_msgs/msg/pose_array.hpp"
#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"
#include "isaac_ros_nitros/nitros_node.hpp"
#include "rclcpp/rclcpp.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dope
{

/**
 * @class DopeDecoderNode
 * @brief This node performs pose estimation of a known object from a single RGB
 *        image
 *        Paper: See https://arxiv.org/abs/1809.10790
 *        Code: https://github.com/NVlabs/Deep_Object_Pose
 */
class DopeDecoderNode : public nitros::NitrosNode
{
public:
  explicit DopeDecoderNode(rclcpp::NodeOptions options = rclcpp::NodeOptions());
  ~DopeDecoderNode();

  void postLoadGraphCallback() override;

private:
  // The name of the YAML configuration file
  const std::string configuration_file_;

  // The class name of the object we're locating
  const std::string object_name_;

  // The dimensions of the cuboid around the object we're locating
  std::vector<double> object_dimensions_;

  // The camera matrix used to capture the input images
  std::vector<double> camera_matrix_;
};

}  // namespace dope
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_DOPE__DOPE_DECODER_NODE_HPP_
