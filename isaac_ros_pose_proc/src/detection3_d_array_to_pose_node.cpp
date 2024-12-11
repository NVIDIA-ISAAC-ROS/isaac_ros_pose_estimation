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
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>

namespace nvidia
{
namespace isaac_ros
{
namespace pose_proc
{

/*
ROS 2 node that selects a single object from a vision_msgs::msg::Detection3DArray
based on desired class ID and confidence
*/
class Detection3DArrayToPoseNode : public rclcpp::Node
{
public:
  explicit Detection3DArrayToPoseNode(const rclcpp::NodeOptions & options)
  : Node("detection3_d_array_to_pose_node", options),
    desired_class_id_(declare_parameter<std::string>("desired_class_id", "")),
    detection3_d_array_sub_{create_subscription<vision_msgs::msg::Detection3DArray>(
        "detection3_d_array_input", 10,
        std::bind(&Detection3DArrayToPoseNode::detection3DArrayCallback, this,
        std::placeholders::_1))},
    pose_pub_{create_publisher<geometry_msgs::msg::PoseStamped>(
        "pose_output", 10)} {}

  void detection3DArrayCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
  {
    if (msg->detections.empty()) {
      return;
    }

    // Find the detection with the highest confidence
    float max_confidence = -1;
    geometry_msgs::msg::PoseStamped max_confidence_pose;
    max_confidence_pose.header = msg->header;
    // Iterate through the detections and find the one with the highest confidence
    for (const auto & detection : msg->detections) {
      // Iterate through all the hypotheses for this detection
      // and find the one with the highest confidence
      for (const auto & result : detection.results) {
        if (result.hypothesis.score > max_confidence && (desired_class_id_.empty() ||
          desired_class_id_ == result.hypothesis.class_id))
        {
          max_confidence = result.hypothesis.score;
          max_confidence_pose.pose = result.pose.pose;
        }
      }
    }
    pose_pub_->publish(max_confidence_pose);
  }

private:
  std::string desired_class_id_;
  rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr detection3_d_array_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
};

}  // namespace pose_proc
}  // namespace isaac_ros
}  // namespace nvidia

// Register the component with the ROS system to create a shared library
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::pose_proc::Detection3DArrayToPoseNode)
