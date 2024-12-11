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

#include <optional>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace nvidia
{
namespace isaac_ros
{
namespace pose_proc
{

class OutlierFilterNode : public rclcpp::Node
{
public:
  explicit OutlierFilterNode(const rclcpp::NodeOptions & options)
  : Node("outlier_filter_node", options),
    distance_threshold_(declare_parameter<double>("distance_threshold", 0.2)),
    angle_threshold_(declare_parameter<double>("angle_threshold", 30.0)),
    pose_stale_time_threshold_(declare_parameter<double>("pose_stale_time_threshold", 5.0)),
    enable_tf_publishing_(declare_parameter<bool>("enable_tf_publishing", false)),
    child_frame_id_(declare_parameter<std::string>("child_frame_id", "")),
    pose_sub_{create_subscription<geometry_msgs::msg::PoseStamped>(
        "pose_input", 10,
        std::bind(&OutlierFilterNode::poseCallback, this, std::placeholders::_1))},
    pose_pub_{create_publisher<geometry_msgs::msg::PoseStamped>(
        "pose_output", 10)},
    tf_broadcaster_{std::make_unique<tf2_ros::TransformBroadcaster>(*this)}
  {
    if (enable_tf_publishing_ && child_frame_id_ == "") {
      RCLCPP_ERROR(
        get_logger(),
        "[OutlierFilterNode] Child frame id must be specified if output_tf is true");
      throw std::invalid_argument(
              "[OutlierFilterNode] Invalid child frame id"
              "Child frame id must be specified if output_tf is true.");
    }
  }

  bool isOutlier(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    // Not outlier if it is the first pose
    if (!pose_to_compare_.has_value()) {
      pose_to_compare_ = *msg;
      return false;
    }

    // Calculate euclidean distance
    const double dx{msg->pose.position.x - pose_to_compare_.value().pose.position.x};
    const double dy{msg->pose.position.y - pose_to_compare_.value().pose.position.y};
    const double dz{msg->pose.position.z - pose_to_compare_.value().pose.position.z};
    const double euclidean_dist{std::sqrt(dx * dx + dy * dy + dz * dz)};

    // Calculate relative angle
    tf2::Quaternion q_current, q_compare;
    tf2::fromMsg(msg->pose.orientation, q_current);
    tf2::fromMsg(pose_to_compare_.value().pose.orientation, q_compare);
    const double angle_diff_rad{q_current.angleShortestPath(q_compare)};
    const double angle_diff_deg{angle_diff_rad * 180.0 / M_PI};

    // Calculate time elapsed since previous good pose
    rclcpp::Time msg_time = rclcpp::Time(msg->header.stamp);
    rclcpp::Time pose_to_compare_time = rclcpp::Time(pose_to_compare_->header.stamp);
    rclcpp::Duration time_since_good_pose = msg_time - pose_to_compare_time;

    // Not outlier if distance and angle within thresholds or previous good pose is too old
    if ((euclidean_dist < distance_threshold_ && angle_diff_deg < angle_threshold_) ||
      (time_since_good_pose.seconds() > pose_stale_time_threshold_))
    {
      pose_to_compare_ = *msg;
      return false;
    }
    return true;
  }

  void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    // Prevent publishing if pose is an outlier
    if (isOutlier(msg)) {
      return;
    }

    // Publish output PoseStamped
    pose_pub_->publish(*msg);

    // Broadcast tf if output_tf parameter is true
    if (enable_tf_publishing_) {
      geometry_msgs::msg::TransformStamped tf;
      tf.header = msg->header;
      tf.child_frame_id = child_frame_id_;
      tf.transform.translation.x = msg->pose.position.x;
      tf.transform.translation.y = msg->pose.position.y;
      tf.transform.translation.z = msg->pose.position.z;
      tf.transform.rotation = msg->pose.orientation;
      tf_broadcaster_->sendTransform(tf);
    }
  }

private:
  double distance_threshold_;
  double angle_threshold_;
  double pose_stale_time_threshold_;
  bool enable_tf_publishing_;
  std::string child_frame_id_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::optional<geometry_msgs::msg::PoseStamped> pose_to_compare_;
};

}  // namespace pose_proc
}  // namespace isaac_ros
}  // namespace nvidia

// Register the component with the ROS system to create a shared library
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::pose_proc::OutlierFilterNode)
