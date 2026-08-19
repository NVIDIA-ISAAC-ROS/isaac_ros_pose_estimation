// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <Eigen/Dense>
#include <memory>
#include <string>
#include <vector>

#include "geometry_msgs/msg/pose_array.hpp"
#include "isaac_ros_nitros/types/nitros_type_message_filter_traits.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/exact_time.h"
#include "message_filters/synchronizer.h"
#include "opencv2/core.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "vision_msgs/msg/detection3_d_array.hpp"

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
class DopeDecoderNode : public rclcpp::Node
{
public:
  explicit DopeDecoderNode(const rclcpp::NodeOptions & options);
  ~DopeDecoderNode();

private:
  void DopeDecoderDetectionCallback(
    const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & tensor_list,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info);
  bool UpdateCameraProperties(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info);

  // The name of the YAML configuration file
  const std::string configuration_file_;

  // The class name of the object we're locating
  const std::string object_name_;

  // The frame name for TF publishing
  const std::string tf_frame_name_;

  // Boolean value indicating option to pubishing to TF tree
  const bool enable_tf_publishing_;

  // The minimum value of a peak in a belief map
  const double map_peak_threshold_;

  // The maximum angle threshold for affinity mapping of corners to centroid
  const double affinity_map_angle_threshold_;

  // Double indicating that dope outputs pose rotated by N degrees along y axis
  const double rotation_y_axis_;

  // Double indicating that dope outputs pose rotated by N degrees along x axis
  const double rotation_x_axis_;

  // Double indicating that dope outputs pose rotated by N degrees along z axis
  const double rotation_z_axis_;

  // The dimensions of the cuboid around the object we're locating
  std::vector<double> object_dimensions_;

  // The camera matrix used to capture the input images
  std::vector<double> camera_matrix_;

  rclcpp::QoS input_qos_;
  rclcpp::QoS output_qos_;
  int16_t input_queue_size_;

  // The transform broadcaster for when TF publishing for poses is enabled
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  // CUDA resources
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;

  // NITROS-aware synchronization of belief-map and adjusted CameraInfo inputs
  ::message_filters::Subscriber<nvidia::isaac_ros::nitros::NitrosTensorList> tensor_sub_;
  ::message_filters::Subscriber<sensor_msgs::msg::CameraInfo> camera_info_sub_;
  using ExactPolicy = ::message_filters::sync_policies::ExactTime<
    nvidia::isaac_ros::nitros::NitrosTensorList,
    sensor_msgs::msg::CameraInfo>;
  ::message_filters::Synchronizer<ExactPolicy> exact_sync_;

  // Publisher for output Detection3DArray messages
  rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr detections_pub_;

  // Parsed parameters
  Eigen::Matrix<double, 3, 9> cuboid_3d_points_;
  cv::Mat cv_camera_matrix_;
};

}  // namespace dope
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_DOPE__DOPE_DECODER_NODE_HPP_
