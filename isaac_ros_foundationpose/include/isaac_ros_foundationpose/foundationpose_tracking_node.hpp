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

#ifndef ISAAC_ROS_FOUNDATIONPOSE__FOUNDATIONPOSE_TRACKING_NODE_HPP_
#define ISAAC_ROS_FOUNDATIONPOSE__FOUNDATIONPOSE_TRACKING_NODE_HPP_

#include <string>
#include <vector>

#include <vision_msgs/msg/detection3_d_array.hpp>

#include "isaac_ros_nitros/nitros_node.hpp"
#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"
#include "rclcpp/rclcpp.hpp"

using StringList = std::vector<std::string>;

namespace nvidia
{
namespace isaac_ros
{
namespace foundationpose
{

/**
 * @class FoundationPoseTrackingNode
 * @brief This node performs tracking use the FoundationPose refine network
 */
class FoundationPoseTrackingNode : public nitros::NitrosNode
{
public:
  explicit FoundationPoseTrackingNode(rclcpp::NodeOptions options = rclcpp::NodeOptions());
  ~FoundationPoseTrackingNode();

  void postLoadGraphCallback() override;

  void FoundationPoseTrackingCallback(const gxf_context_t context, nitros::NitrosTypeBase & msg);

private:
  // The name of the YAML configuration file
  const std::string configuration_file_;
  uint32_t resized_image_width_;
  uint32_t resized_image_height_;
  float refine_crop_ratio_;
  float rot_normalizer_;

  // Path to the mesh files
  const std::string mesh_file_path_;

  const float min_depth_;
  const float max_depth_;

  // Models file path
  const std::string refine_model_file_path_;
  const std::string refine_engine_file_path_;

  // Input tensor names
  const StringList refine_input_tensor_names_;
  const StringList refine_input_binding_names_;

  // Output tensor names
  const StringList refine_output_tensor_names_;
  const StringList refine_output_binding_names_;

  const std::string tf_frame_name_;
};

}  // namespace foundationpose
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_FOUNDATIONPOSE__FOUNDATIONPOSE_TRACKING_NODE_HPP_
