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

#include <deque>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

namespace nvidia
{
namespace isaac_ros
{
namespace pose_proc
{
namespace
{
struct position_t
{
  double x{0};
  double y{0};
  double z{0};
};

struct orientation_t
{
  double x{0};
  double y{0};
  double z{0};
  double w{0};
};
}  // namespace

class AveragingFilterNode : public rclcpp::Node
{
public:
  explicit AveragingFilterNode(const rclcpp::NodeOptions & options)
  : Node("averaging_filter_node", options),
    num_samples_(declare_parameter<int>("num_samples", 10)),
    enable_tf_publishing_(declare_parameter<bool>("enable_tf_publishing", false)),
    child_frame_id_(declare_parameter<std::string>("child_frame_id", "")),
    pose_sub_{create_subscription<geometry_msgs::msg::PoseStamped>(
        "pose_input", 10,
        std::bind(&AveragingFilterNode::poseCallback, this, std::placeholders::_1))},
    pose_pub_{create_publisher<geometry_msgs::msg::PoseStamped>(
        "pose_output", 10)},
    tf_broadcaster_{std::make_unique<tf2_ros::TransformBroadcaster>(*this)}
  {
    if (enable_tf_publishing_ && child_frame_id_ == "") {
      RCLCPP_ERROR(
        get_logger(),
        "[AveragingFilterNode] Child frame id must be specified if output_tf is true");
      throw std::invalid_argument(
              "[AveragingFilterNode] Invalid child frame id"
              "Child frame id must be specified if output_tf is true.");
    }
  }

  void removeOldestPose()
  {
    if (pose_history_.size() >= static_cast<size_t>(num_samples_)) {
      position_total_.x -= pose_history_.front()->pose.position.x;
      position_total_.y -= pose_history_.front()->pose.position.y;
      position_total_.z -= pose_history_.front()->pose.position.z;
      orientation_total_.x -= pose_history_.front()->pose.orientation.x;
      orientation_total_.y -= pose_history_.front()->pose.orientation.y;
      orientation_total_.z -= pose_history_.front()->pose.orientation.z;
      orientation_total_.w -= pose_history_.front()->pose.orientation.w;
      pose_history_.pop_front();
    }
  }

  void addLatestPose(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    pose_history_.push_back(msg);
    position_total_.x += msg->pose.position.x;
    position_total_.y += msg->pose.position.y;
    position_total_.z += msg->pose.position.z;
    orientation_total_.x += msg->pose.orientation.x;
    orientation_total_.y += msg->pose.orientation.y;
    orientation_total_.z += msg->pose.orientation.z;
    orientation_total_.w += msg->pose.orientation.w;
  }

  geometry_msgs::msg::PoseStamped getAveragePose(geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    geometry_msgs::msg::PoseStamped averaged_pose;
    averaged_pose.header = msg->header;
    averaged_pose.pose.position.x = position_total_.x / pose_history_.size();
    averaged_pose.pose.position.y = position_total_.y / pose_history_.size();
    averaged_pose.pose.position.z = position_total_.z / pose_history_.size();
    const double quat_distance{std::sqrt(
        orientation_total_.x * orientation_total_.x +
        orientation_total_.y * orientation_total_.y +
        orientation_total_.z * orientation_total_.z +
        orientation_total_.w * orientation_total_.w)};
    averaged_pose.pose.orientation.x = orientation_total_.x / quat_distance;
    averaged_pose.pose.orientation.y = orientation_total_.y / quat_distance;
    averaged_pose.pose.orientation.z = orientation_total_.z / quat_distance;
    averaged_pose.pose.orientation.w = orientation_total_.w / quat_distance;
    return averaged_pose;
  }

  void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    removeOldestPose();
    addLatestPose(msg);
    if (pose_history_.size() < static_cast<size_t>(num_samples_)) {
      return;
    }

    geometry_msgs::msg::PoseStamped out_pose = getAveragePose(msg);

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
  int num_samples_;
  bool enable_tf_publishing_;
  std::string child_frame_id_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::deque<geometry_msgs::msg::PoseStamped::SharedPtr> pose_history_;
  position_t position_total_;
  orientation_t orientation_total_;
};

}  // namespace pose_proc
}  // namespace isaac_ros
}  // namespace nvidia

// Register the component with the ROS system to create a shared library
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::pose_proc::AveragingFilterNode)
