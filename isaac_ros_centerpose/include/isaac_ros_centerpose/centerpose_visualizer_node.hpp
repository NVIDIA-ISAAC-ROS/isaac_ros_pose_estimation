// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "rclcpp/rclcpp.hpp"
#include "Eigen/Dense"

#include "isaac_ros_centerpose/centerpose_types.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include "vision_msgs/msg/detection3_d.hpp"
#include "vision_msgs/msg/object_hypothesis_with_pose.hpp"
#include "vision_msgs/msg/bounding_box3_d.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "message_filters/subscriber.hpp"
#include "message_filters/synchronizer.hpp"
#include "message_filters/sync_policies/exact_time.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace centerpose
{

class CenterPoseVisualizerNode : public rclcpp::Node
{
public:
  explicit CenterPoseVisualizerNode(const rclcpp::NodeOptions & options);
  ~CenterPoseVisualizerNode() = default;

private:
  void InputCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const vision_msgs::msg::Detection3DArray::ConstSharedPtr & detection3darray,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info
  );
  // Input Parameters
  bool show_axes_;
  int32_t bounding_box_color_;
  int16_t input_queue_size_;
  int16_t output_queue_size_;

  // Subscriptions and publishers
  message_filters::Subscriber<sensor_msgs::msg::Image> image_sub_;
  message_filters::Subscriber<vision_msgs::msg::Detection3DArray> detection3darray_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> camera_info_sub_;

  using ExactPolicy = message_filters::sync_policies::ExactTime<
    sensor_msgs::msg::Image,
    vision_msgs::msg::Detection3DArray,
    sensor_msgs::msg::CameraInfo
  >;
  message_filters::Synchronizer<ExactPolicy> image_camera_info_sync_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;

  // CUDA resources
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;
};

}  // namespace centerpose
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_CENTERPOSE__CENTERPOSE_VISUALIZER_NODE_HPP_
