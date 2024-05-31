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

#ifndef ISAAC_ROS_FOUNDATIONPOSE__FOUNDATIONPOSE_NODE_HPP_
#define ISAAC_ROS_FOUNDATIONPOSE__FOUNDATIONPOSE_NODE_HPP_

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
 * @class FoundationPoseNode
 * @brief This node performs pose estimation of an unseen object from RGBD
 */
class FoundationPoseNode : public nitros::NitrosNode
{
public:
  explicit FoundationPoseNode(rclcpp::NodeOptions options = rclcpp::NodeOptions());
  ~FoundationPoseNode();

  void postLoadGraphCallback() override;

  void FoundationPoseDetectionCallback(const gxf_context_t context, nitros::NitrosTypeBase & msg);

private:
  // The name of the YAML configuration file
  const std::string configuration_file_;
  uint32_t max_hypothesis_;
  uint32_t resized_image_width_;
  uint32_t resized_image_height_;
  float refine_crop_ratio_;
  float score_crop_ratio_;
  float rot_normalizer_;

  // Path to the mesh files
  const std::string mesh_file_path_;
  const std::string texture_path_;

  const float min_depth_;
  const float max_depth_;
  const int32_t refine_iterations_;

  // Models file path
  const std::string refine_model_file_path_;
  const std::string refine_engine_file_path_;
  const std::string score_model_file_path_;
  const std::string score_engine_file_path_;

  // Input tensor names
  const StringList refine_input_tensor_names_;
  const StringList refine_input_binding_names_;
  const StringList score_input_tensor_names_;
  const StringList score_input_binding_names_;

  // Output tensor names
  const StringList refine_output_tensor_names_;
  const StringList refine_output_binding_names_;
  const StringList score_output_tensor_names_;
  const StringList score_output_binding_names_;

  const std::string tf_frame_name_;
};

}  // namespace foundationpose
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_FOUNDATIONPOSE__FOUNDATIONPOSE_NODE_HPP_
