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

#include <tf2_ros/transform_broadcaster.h>

#include <limits>
#include <optional>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

namespace nvidia
{
namespace isaac_ros
{
namespace pose_proc
{

class DistanceSelectorNode : public rclcpp::Node
{
public:
  explicit DistanceSelectorNode(const rclcpp::NodeOptions & options)
  : Node("distance_selector_node", options),
    enable_tf_publishing_(declare_parameter<bool>("enable_tf_publishing", false)),
    child_frame_id_(declare_parameter<std::string>("child_frame_id", "")),
    pose_sub_{create_subscription<geometry_msgs::msg::PoseStamped>(
        "pose_input", 10,
        std::bind(&DistanceSelectorNode::poseCallback, this, std::placeholders::_1))},
    pose_array_sub_{create_subscription<geometry_msgs::msg::PoseArray>(
        "pose_array_input", 10,
        std::bind(&DistanceSelectorNode::poseArrayCallback, this, std::placeholders::_1))},
    pose_pub_{create_publisher<geometry_msgs::msg::PoseStamped>(
        "pose_output", 10)},
    tf_broadcaster_{std::make_unique<tf2_ros::TransformBroadcaster>(*this)}
  {
    if (enable_tf_publishing_ && child_frame_id_ == "") {
      RCLCPP_ERROR(
        get_logger(),
        "[DistanceSelectorNode] Child frame id must be specified if output_tf is true");
      throw std::invalid_argument(
              "[DistanceSelectorNode] Invalid child frame id"
              "Child frame id must be specified if output_tf is true.");
    }
  }

  geometry_msgs::msg::PoseStamped distanceFilter(const geometry_msgs::msg::PoseArray::SharedPtr msg)
  {
    geometry_msgs::msg::PoseStamped closest_pose;
    closest_pose.header = msg->header;

    // Calculate closest pose based on position
    double closest_distance = std::numeric_limits<double>::max();
    for (const geometry_msgs::msg::Pose & pose : msg->poses) {
      const double dx{pose_to_compare_.value().position.x - pose.position.x};
      const double dy{pose_to_compare_.value().position.y - pose.position.y};
      const double dz{pose_to_compare_.value().position.z - pose.position.z};
      const double curr_dist{std::sqrt(dx * dx + dy * dy + dz * dz)};
      if (curr_dist < closest_distance) {
        closest_distance = curr_dist;
        closest_pose.pose = pose;
      }
    }

    return closest_pose;
  }

  void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    pose_to_compare_ = msg->pose;
  }

  void poseArrayCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
  {
    // Ensure pose to compare has been received
    if (!pose_to_compare_.has_value()) {
      RCLCPP_WARN(get_logger(), "Pose to compare has not been received.");
      return;
    }

    // Find pose that is closest to pose_to_compare
    geometry_msgs::msg::PoseStamped out_pose = distanceFilter(msg);

    // Publish output PoseStamped
    pose_pub_->publish(out_pose);

    // Broadcast tf if output_tf parameter is true
    if (enable_tf_publishing_) {
      geometry_msgs::msg::TransformStamped tf;
      tf.header = out_pose.header;
      tf.child_frame_id = child_frame_id_;
      tf.transform.translation.x = out_pose.pose.position.x;
      tf.transform.translation.y = out_pose.pose.position.y;
      tf.transform.translation.z = out_pose.pose.position.z;
      tf.transform.rotation = out_pose.pose.orientation;
      tf_broadcaster_->sendTransform(tf);
    }
  }

private:
  bool enable_tf_publishing_;
  std::string child_frame_id_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr pose_array_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  std::optional<geometry_msgs::msg::Pose> pose_to_compare_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

}  // namespace pose_proc
}  // namespace isaac_ros
}  // namespace nvidia

// Register the component with the ROS system to create a shared library
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::pose_proc::DistanceSelectorNode)
