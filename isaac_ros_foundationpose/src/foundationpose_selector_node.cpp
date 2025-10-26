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

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/exact_time.h"

#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"

#include "isaac_ros_managed_nitros/managed_nitros_publisher.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_message_filters_subscriber.hpp"

#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_image_type/nitros_image_view.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace foundationpose
{

/*
ROS 2 node that select the next action - tracking or pose estimation.
State flow: kPoseEstimatino -> kWaitingRest -> kTracking
*/

class Selector : public rclcpp::Node
{
public:
  explicit Selector(const rclcpp::NodeOptions & options)
  : Node("selector", options)
  {
    // Create publishers for pose estimation
    pose_estimation_image_pub_ = std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
          nvidia::isaac_ros::nitros::NitrosImage>>(
      this, "pose_estimation/image",
      nvidia::isaac_ros::nitros::nitros_image_rgb8_t::supported_type_name);
    pose_estimation_depth_pub_ = std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
          nvidia::isaac_ros::nitros::NitrosImage>>(
      this, "pose_estimation/depth_image",
      nvidia::isaac_ros::nitros::nitros_image_32FC1_t::supported_type_name);
    pose_estimation_segmenation_pub_ = std::make_shared<
      nvidia::isaac_ros::nitros::ManagedNitrosPublisher<nvidia::isaac_ros::nitros::NitrosImage>>(
      this, "pose_estimation/segmentation",
      nvidia::isaac_ros::nitros::nitros_image_mono8_t::supported_type_name);
    pose_estimation_camera_pub_ = this->create_publisher<
      sensor_msgs::msg::CameraInfo>("pose_estimation/camera_info", 1);

    // Create publishers for tracking
    tracking_image_pub_ = std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
          nvidia::isaac_ros::nitros::NitrosImage>>(
      this, "tracking/image",
      nvidia::isaac_ros::nitros::nitros_image_rgb8_t::supported_type_name);
    tracking_depth_pub_ = std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
          nvidia::isaac_ros::nitros::NitrosImage>>(
      this, "tracking/depth_image",
      nvidia::isaac_ros::nitros::nitros_image_32FC1_t::supported_type_name);
    tracking_pose_pub_ = this->create_publisher<
      isaac_ros_tensor_list_interfaces::msg::TensorList>("tracking/pose_input", 1);
    tracking_camera_pub_ = this->create_publisher<
      sensor_msgs::msg::CameraInfo>("tracking/camera_info", 1);

    // Exact Sync
    using namespace std::placeholders;
    exact_sync_ = std::make_shared<ExactSync>(
      ExactPolicy(20), rgb_image_sub_, depth_image_sub_, segmentation_sub_,
      camera_info_sub_);
    exact_sync_->registerCallback(
      std::bind(&Selector::selectionCallback, this, _1, _2, _3, _4));

    segmentation_sub_.subscribe(this, "segmentation");
    rgb_image_sub_.subscribe(this, "image");
    depth_image_sub_.subscribe(this, "depth_image");
    camera_info_sub_.subscribe(this, "camera_info");

    // Create subscriber for pose input
    tracking_output_sub_ =
      this->create_subscription<isaac_ros_tensor_list_interfaces::msg::TensorList>(
      "tracking/pose_matrix_output", 1, std::bind(&Selector::poseForwardCallback, this, _1));
    pose_estimation_output_sub_ =
      this->create_subscription<isaac_ros_tensor_list_interfaces::msg::TensorList>(
      "pose_estimation/pose_matrix_output", 1, std::bind(&Selector::poseResetCallback, this, _1));

    // reset period in ms after which pose estimation will be triggered
    this->declare_parameter<int>("reset_period", 20000);
    this->get_parameter("reset_period", reset_period_);

    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(reset_period_),
      std::bind(&Selector::timerCallback, this));
  }

  void selectionCallback(
    const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & image_msg,
    const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & depth_msg,
    const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & segmentaion_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    // Trigger next action
    if (state_ == State::kPoseEstimation) {
      // Publish all other messages except pose matrix to pose estimation
      pose_estimation_image_pub_->publish(*image_msg);
      pose_estimation_camera_pub_->publish(*camera_info_msg);
      pose_estimation_depth_pub_->publish(*depth_msg);
      pose_estimation_segmenation_pub_->publish(*segmentaion_msg);
      state_ = State::kWaitingReset;
    } else if (state_ == State::kTracking) {
      // Publish all messages except segmentation to tracking
      tracking_image_pub_->publish(*image_msg);
      tracking_camera_pub_->publish(*camera_info_msg);
      tracking_depth_pub_->publish(*depth_msg);
      tracking_pose_pub_->publish(*tracking_pose_msg_);
    }
  }

  void poseForwardCallback(
    const isaac_ros_tensor_list_interfaces::msg::TensorList::ConstSharedPtr & tracking_output_msg)
  {
    std::unique_lock<std::mutex> lock(pose_mutex_);
    // Discard the stale pose messages from tracking to avoid drift
    if (state_ == State::kTracking) {
      tracking_pose_msg_ = tracking_output_msg;
    }
  }

  void poseResetCallback(
    const isaac_ros_tensor_list_interfaces::msg::TensorList::ConstSharedPtr
    & pose_estimation_output_msg)
  {
    std::unique_lock<std::mutex> lock(pose_mutex_);
    tracking_pose_msg_ = pose_estimation_output_msg;
    state_ = kTracking;
  }

  void timerCallback()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    state_ = State::kPoseEstimation;
  }

private:
  // Publishers for pose estimation
  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
      nvidia::isaac_ros::nitros::NitrosImage>> pose_estimation_image_pub_;
  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
      nvidia::isaac_ros::nitros::NitrosImage>> pose_estimation_depth_pub_;
  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
      nvidia::isaac_ros::nitros::NitrosImage>> pose_estimation_segmenation_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr pose_estimation_camera_pub_;

  // Publishers for tracking
  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
      nvidia::isaac_ros::nitros::NitrosImage>> tracking_image_pub_;
  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
      nvidia::isaac_ros::nitros::NitrosImage>> tracking_depth_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr tracking_camera_pub_;
  rclcpp::Publisher<isaac_ros_tensor_list_interfaces::msg::TensorList>::SharedPtr
    tracking_pose_pub_;

  // Subscribers
  nvidia::isaac_ros::nitros::message_filters::Subscriber<
    nvidia::isaac_ros::nitros::NitrosImageView> rgb_image_sub_;
  nvidia::isaac_ros::nitros::message_filters::Subscriber<
    nvidia::isaac_ros::nitros::NitrosImageView> depth_image_sub_;
  nvidia::isaac_ros::nitros::message_filters::Subscriber<
    nvidia::isaac_ros::nitros::NitrosImageView> segmentation_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> camera_info_sub_;

  rclcpp::Subscription<isaac_ros_tensor_list_interfaces::msg::TensorList>::SharedPtr
    tracking_output_sub_;
  rclcpp::Subscription<isaac_ros_tensor_list_interfaces::msg::TensorList>::SharedPtr
    pose_estimation_output_sub_;

  enum State
  {
    kTracking,
    kPoseEstimation,
    kWaitingReset
  };
  // State
  State state_ = State::kPoseEstimation;

  // Exact message sync policy
  using ExactPolicy = message_filters::sync_policies::ExactTime<
    nvidia::isaac_ros::nitros::NitrosImage,
    nvidia::isaac_ros::nitros::NitrosImage,
    nvidia::isaac_ros::nitros::NitrosImage,
    sensor_msgs::msg::CameraInfo>;
  using ExactSync = message_filters::Synchronizer<ExactPolicy>;
  std::shared_ptr<ExactSync> exact_sync_;

  rclcpp::TimerBase::SharedPtr timer_;
  std::mutex mutex_;
  std::mutex pose_mutex_;
  isaac_ros_tensor_list_interfaces::msg::TensorList::ConstSharedPtr tracking_pose_msg_;

  int reset_period_;
};

}  // namespace foundationpose
}  // namespace isaac_ros
}  // namespace nvidia

// Register the component with the ROS system to create a shared library
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::foundationpose::Selector)
