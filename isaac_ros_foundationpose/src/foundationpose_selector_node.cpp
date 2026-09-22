// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_srvs/srv/trigger.hpp>

#include "message_filters/subscriber.hpp"
#include "message_filters/synchronizer.hpp"
#include "message_filters/sync_policies/exact_time.hpp"

#include "isaac_ros_tensor_msgs/tensor_list_msg.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace foundationpose
{

namespace TensorMsg = nvidia::isaac_ros::isaac_ros_tensor_msgs;

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
    rclcpp::PublisherOptions image_pub_options;
    image_pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

    // Pose-estimation outputs
    pose_estimation_image_pub_ =
      create_publisher<sensor_msgs::msg::Image>(
      "pose_estimation/image", 1, image_pub_options);
    pose_estimation_depth_pub_ =
      create_publisher<sensor_msgs::msg::Image>(
      "pose_estimation/depth_image", 1, image_pub_options);
    pose_estimation_segmenation_pub_ =
      create_publisher<sensor_msgs::msg::Image>(
      "pose_estimation/segmentation", 1, image_pub_options);
    pose_estimation_camera_pub_ =
      create_publisher<sensor_msgs::msg::CameraInfo>("pose_estimation/camera_info", 1);

    // Tracking outputs
    tracking_image_pub_ =
      create_publisher<sensor_msgs::msg::Image>("tracking/image", 1, image_pub_options);
    tracking_depth_pub_ =
      create_publisher<sensor_msgs::msg::Image>("tracking/depth_image", 1, image_pub_options);
    {
      rclcpp::QoS pose_qos(1);
      pose_qos.transient_local();
      rclcpp::PublisherOptions tensor_pub_options;
      tensor_pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
      tracking_pose_pub_ = create_publisher<
        TensorMsg::TensorListMsg>("tracking/pose_input", pose_qos, tensor_pub_options);
    }
    tracking_camera_pub_ =
      create_publisher<sensor_msgs::msg::CameraInfo>("tracking/camera_info", 1);

    // RT-DETR (gated; only forwarded while in kPoseEstimation)
    rt_detr_image_pub_ =
      create_publisher<sensor_msgs::msg::Image>("rt_detr/image", 1, image_pub_options);
    rt_detr_camera_info_pub_ =
      create_publisher<sensor_msgs::msg::CameraInfo>("rt_detr/camera_info", 1);

    // Subscribers
    using namespace std::placeholders;
    const rclcpp::QoS input_qos(10);
    rclcpp::SubscriptionOptions image_sub_options;
    image_sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
    image_sub_options.acceptable_buffer_backends = "any";
    rgb_image_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>();
    rgb_image_sub_->subscribe(this, "image", input_qos, image_sub_options);
    depth_image_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>();
    depth_image_sub_->subscribe(this, "depth_image", input_qos, image_sub_options);
    segmentation_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>();
    segmentation_sub_->subscribe(this, "segmentation", input_qos, image_sub_options);
    camera_info_sub_ =
      std::make_shared<message_filters::Subscriber<sensor_msgs::msg::CameraInfo>>();
    camera_info_sub_->subscribe(this, "camera_info", input_qos);

    // Pose-estimation sync (rgb + depth + segmentation + camera_info, runs at RT-DETR rate)
    exact_sync_ = std::make_shared<ExactSync>(
      ExactPolicy(20), *rgb_image_sub_, *depth_image_sub_, *segmentation_sub_,
      *camera_info_sub_);
    exact_sync_->registerCallback(
      std::bind(&Selector::selectionCallback, this, _1, _2, _3, _4));

    // Tracking sync (rgb + depth + camera_info, runs at full input rate)
    tracking_sync_ = std::make_shared<TrackingSync>(
      TrackingPolicy(20), *rgb_image_sub_, *depth_image_sub_, *camera_info_sub_);
    tracking_sync_->registerCallback(
      std::bind(&Selector::trackingCallback, this, _1, _2, _3));

    rgb_image_sub_->registerCallback(
      std::bind(&Selector::forwardImageToDetection, this, _1));
    camera_info_sub_->registerCallback(
      std::bind(&Selector::forwardCameraInfoToDetection, this, _1));

    rclcpp::SubscriptionOptions tensor_sub_options;
    tensor_sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
    tensor_sub_options.acceptable_buffer_backends = "any";
    tracking_output_sub_ =
      this->create_subscription<TensorMsg::TensorListMsg>(
      "tracking/pose_matrix_output", 1, std::bind(&Selector::poseForwardCallback, this, _1),
      tensor_sub_options);
    pose_estimation_output_sub_ =
      this->create_subscription<TensorMsg::TensorListMsg>(
      "pose_estimation/pose_matrix_output", 1, std::bind(&Selector::poseResetCallback, this, _1),
      tensor_sub_options);
    reset_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "selector/reset", std::bind(&Selector::trackingResetCallback, this, _1, _2));

    this->declare_parameter<int>("reset_period", 600000);
    this->get_parameter("reset_period", reset_period_);

    this->declare_parameter<int>("tracking_timeout_ms", 1000);
    this->get_parameter("tracking_timeout_ms", tracking_timeout_ms_);

    last_tracking_msg_time_ = this->get_clock()->now();

    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(reset_period_),
      std::bind(&Selector::timerCallback, this));

    health_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(200),
      std::bind(&Selector::healthCheck, this));
  }

  void selectionCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr & image_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr & segmentaion_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (state_ != State::kPoseEstimation) {
      return;
    }
    pose_estimation_image_pub_->publish(*image_msg);
    pose_estimation_camera_pub_->publish(*camera_info_msg);
    pose_estimation_depth_pub_->publish(*depth_msg);
    pose_estimation_segmenation_pub_->publish(*segmentaion_msg);
    state_ = State::kWaitingReset;
    RCLCPP_INFO(this->get_logger(),
      "[selector] dispatched pose-estimation frame, state->kWaitingReset");
  }

  void trackingCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr & image_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (state_ != State::kTracking) {
      return;
    }
    tracking_image_pub_->publish(*image_msg);
    tracking_camera_pub_->publish(*camera_info_msg);
    tracking_depth_pub_->publish(*depth_msg);
  }

  void forwardImageToDetection(
    const sensor_msgs::msg::Image::ConstSharedPtr & msg)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (state_ != State::kPoseEstimation) {
      return;
    }
    rt_detr_image_pub_->publish(*msg);
  }

  void forwardCameraInfoToDetection(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (state_ != State::kPoseEstimation) {
      return;
    }
    rt_detr_camera_info_pub_->publish(*msg);
  }

  void poseForwardCallback(
    const TensorMsg::TensorListMsg::ConstSharedPtr & tracking_output_msg)
  {
    last_tracking_msg_time_ = this->get_clock()->now();
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (state_ != State::kTracking) {
        return;
      }
    }
    tracking_pose_pub_->publish(*tracking_output_msg);
  }

  void poseResetCallback(
    const TensorMsg::TensorListMsg::ConstSharedPtr & pose_estimation_output_msg)
  {
    last_tracking_msg_time_ = this->get_clock()->now();
    {
      std::unique_lock<std::mutex> lock(mutex_);
      RCLCPP_INFO(this->get_logger(), "[selector] pose_estimation pose received, state->kTracking");
      state_ = kTracking;
    }
    tracking_pose_pub_->publish(*pose_estimation_output_msg);
  }

  void trackingResetCallback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    RCLCPP_INFO(this->get_logger(), "[selector] tracking reset requested, state->kPoseEstimation");
    state_ = State::kPoseEstimation;
    last_tracking_msg_time_ = this->get_clock()->now();
    response->success = true;
    response->message = "selector reset to pose estimation";
  }

  void timerCallback()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (state_ == State::kTracking) {
      RCLCPP_INFO(this->get_logger(),
        "[selector] periodic timer fired, state->kPoseEstimation");
      state_ = State::kPoseEstimation;
    }
  }

  void healthCheck()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (state_ == State::kPoseEstimation) {
      return;
    }
    const auto elapsed_ms =
      (this->get_clock()->now() - last_tracking_msg_time_).nanoseconds() / 1000000;
    if (elapsed_ms > tracking_timeout_ms_) {
      RCLCPP_WARN(this->get_logger(),
        "[selector] no pose output for %ld ms (state=%d), state->kPoseEstimation",
        elapsed_ms, static_cast<int>(state_));
      state_ = State::kPoseEstimation;
    }
  }

private:
  // Pose-estimation publishers
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pose_estimation_image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pose_estimation_depth_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr
    pose_estimation_segmenation_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr pose_estimation_camera_pub_;

  // Tracking publishers
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr tracking_image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr tracking_depth_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr tracking_camera_pub_;
  rclcpp::Publisher<TensorMsg::TensorListMsg>::SharedPtr tracking_pose_pub_;

  // RT-DETR (gated)
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rt_detr_image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr rt_detr_camera_info_pub_;

  // Subscribers
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> rgb_image_sub_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> depth_image_sub_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> segmentation_sub_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::CameraInfo>> camera_info_sub_;

  rclcpp::Subscription<TensorMsg::TensorListMsg>::SharedPtr tracking_output_sub_;
  rclcpp::Subscription<TensorMsg::TensorListMsg>::SharedPtr pose_estimation_output_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_srv_;

  enum State
  {
    kTracking,
    kPoseEstimation,
    kWaitingReset
  };
  // State
  State state_ = State::kPoseEstimation;

  // Pose-estimation sync (rgb, depth, segmentation, camera_info)
  using ExactPolicy = message_filters::sync_policies::ExactTime<
    sensor_msgs::msg::Image,
    sensor_msgs::msg::Image,
    sensor_msgs::msg::Image,
    sensor_msgs::msg::CameraInfo>;
  using ExactSync = message_filters::Synchronizer<ExactPolicy>;
  std::shared_ptr<ExactSync> exact_sync_;

  // Tracking sync (rgb, depth, camera_info - no segmentation)
  using TrackingPolicy = message_filters::sync_policies::ExactTime<
    sensor_msgs::msg::Image,
    sensor_msgs::msg::Image,
    sensor_msgs::msg::CameraInfo>;
  using TrackingSync = message_filters::Synchronizer<TrackingPolicy>;
  std::shared_ptr<TrackingSync> tracking_sync_;

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr health_timer_;
  std::mutex mutex_;
  rclcpp::Time last_tracking_msg_time_;

  int reset_period_;
  int tracking_timeout_ms_;
};

}  // namespace foundationpose
}  // namespace isaac_ros
}  // namespace nvidia

// Register the component with the ROS system to create a shared library
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::foundationpose::Selector)
