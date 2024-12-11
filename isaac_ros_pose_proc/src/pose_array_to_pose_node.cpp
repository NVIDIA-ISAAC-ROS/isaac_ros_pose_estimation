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

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

namespace nvidia
{
namespace isaac_ros
{
namespace pose_proc
{

// This node only selects the first pose out of the input pose array and throws away all the rest
class PoseArrayToPoseNode : public rclcpp::Node
{
public:
  explicit PoseArrayToPoseNode(const rclcpp::NodeOptions & options)
  : Node("pose_array_to_pose_node", options),
    pose_array_sub_{create_subscription<geometry_msgs::msg::PoseArray>(
        "pose_array_input", 10,
        std::bind(&PoseArrayToPoseNode::poseArrayCallback, this, std::placeholders::_1))},
    pose_pub_{create_publisher<geometry_msgs::msg::PoseStamped>(
        "pose_output", 10)} {}

  void poseArrayCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
  {
    if (!msg->poses.empty()) {
      geometry_msgs::msg::PoseStamped out_msg;
      out_msg.header = msg->header;
      out_msg.pose = msg->poses[0];
      pose_pub_->publish(out_msg);
    }
  }

private:
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr pose_array_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
};

}  // namespace pose_proc
}  // namespace isaac_ros
}  // namespace nvidia

// Register the component with the ROS system to create a shared library
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::pose_proc::PoseArrayToPoseNode)
