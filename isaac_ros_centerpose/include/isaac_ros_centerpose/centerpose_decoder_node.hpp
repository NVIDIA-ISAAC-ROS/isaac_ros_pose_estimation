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

#ifndef ISAAC_ROS_CENTERPOSE__CENTERPOSE_DECODER_NODE_HPP_
#define ISAAC_ROS_CENTERPOSE__CENTERPOSE_DECODER_NODE_HPP_

#include <string>
#include <vector>

#include "Eigen/Dense"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_centerpose/centerpose_detection.hpp"
#include "isaac_ros_tensor_msgs/msg/tensor_list.hpp"
#include "message_filters/subscriber.hpp"
#include "message_filters/synchronizer.hpp"
#include "message_filters/sync_policies/exact_time.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "tensor_msgs/msg/experimental_tensor.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace centerpose
{

using Tensor = tensor_msgs::msg::ExperimentalTensor;
using TensorList = isaac_ros_tensor_msgs::msg::TensorList;

/**
 * @class CenterPoseDecoderNode
 * @brief This node performs pose estimation of a known category from a single RGB image
 *
 */
class CenterPoseDecoderNode : public rclcpp::Node
{
public:
  explicit CenterPoseDecoderNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~CenterPoseDecoderNode() = default;

private:
  void InputCallback(
    const TensorList::ConstSharedPtr & tensor_list,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info
  );
  bool initialize();
  bool UpdateCameraProperties(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info);
  CenterPoseDetectionList ProcessTensor(
    const std::vector<Eigen::MatrixXfRM> & tensors);

  // Input Parameters
  // 2D keypoint decoding size. Width and then height.
  std::vector<int64_t> output_field_size_;

  // Scaling factor for cuboid
  double cuboid_scaling_factor_;

  // Score threshold
  double score_threshold_;

  // Object / instance name that is detected
  std::string object_name_;

  int16_t input_queue_size_;
  int16_t output_queue_size_;

  // Subscriptions and publishers
  message_filters::Subscriber<TensorList> tensor_list_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> camera_info_sub_;

  using ExactPolicy = message_filters::sync_policies::ExactTime<
    TensorList,
    sensor_msgs::msg::CameraInfo
  >;
  message_filters::Synchronizer<ExactPolicy> camera_image_sync_;

  rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr detection3darray_pub_;

  // CUDA resources
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;

  Eigen::Matrix3f camera_matrix_;
  Eigen::Vector2i original_image_size_;
  Eigen::Matrix3fRM affine_transform_;
};

}  // namespace centerpose
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_CENTERPOSE__CENTERPOSE_DECODER_NODE_HPP_
