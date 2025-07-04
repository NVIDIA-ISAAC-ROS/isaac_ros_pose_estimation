// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <mutex>
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

  void UpdateTfFrameNameCallback(const rclcpp::Parameter & param);
  void UpdateMeshFilePathCallback(const rclcpp::Parameter & param);
  void UpdateFixedTranslationsCallback(const rclcpp::Parameter & param);
  void UpdateFixedAxisAnglesCallback(const rclcpp::Parameter & param);
  void UpdateSymmetryAxesCallback(const rclcpp::Parameter & param);
  void UpdateSymmetryPlanesCallback(const rclcpp::Parameter & param);

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
  std::string mesh_file_path_;

  const float min_depth_;
  const float max_depth_;
  const int32_t refine_iterations_;

  // Constraint parameters that can be dynamically updated
  StringList symmetry_axes_;
  StringList symmetry_planes_;
  StringList fixed_axis_angles_;
  StringList fixed_translations_;

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

  // Parameter callback
  std::shared_ptr<rclcpp::ParameterEventHandler> param_subscriber_;
  std::shared_ptr<rclcpp::ParameterCallbackHandle> mesh_file_path_param_cb_handle_;
  std::shared_ptr<rclcpp::ParameterCallbackHandle> tf_frame_name_param_cb_handle_;

  // Callback handles for constraint parameters
  std::shared_ptr<rclcpp::ParameterCallbackHandle> fixed_translations_param_cb_handle_;
  std::shared_ptr<rclcpp::ParameterCallbackHandle> fixed_axis_angles_param_cb_handle_;
  std::shared_ptr<rclcpp::ParameterCallbackHandle> symmetry_axes_param_cb_handle_;
  std::shared_ptr<rclcpp::ParameterCallbackHandle> symmetry_planes_param_cb_handle_;

  // Sync parameters
  int64_t discard_time_ms_;
  bool discard_old_messages_;
  int64_t pose_estimation_timeout_ms_;
  int64_t sync_threshold_;

  // TF frame name
  std::string tf_frame_name_;

  // Debug mode
  bool debug_;
  std::string debug_dir_;

  // Mutex to protect access to tf_frame_name_
  std::mutex mutex_;
};

}  // namespace foundationpose
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_FOUNDATIONPOSE__FOUNDATIONPOSE_NODE_HPP_
