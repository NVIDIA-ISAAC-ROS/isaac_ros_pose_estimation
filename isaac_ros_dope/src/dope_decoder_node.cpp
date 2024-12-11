// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_dope/dope_decoder_node.hpp"

#if __GNUC__ < 9
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#include <string>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "detection3_d_array_message/detection3_d_array_message.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "gxf/std/tensor.hpp"
#include "isaac_ros_nitros_camera_info_type/nitros_camera_info.hpp"
#include "isaac_ros_nitros_detection3_d_array_type/nitros_detection3_d_array.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/transform_broadcaster.h"

namespace nvidia
{
namespace isaac_ros
{
namespace dope
{

using nvidia::gxf::optimizer::GraphIOGroupSupportedDataTypesInfoList;

constexpr char INPUT_COMPONENT_KEY[] = "dope_decoder/tensorlist_in";
constexpr char INPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_tensor_list_nchw_rgb_f32";
constexpr char INPUT_TOPIC_NAME[] = "belief_map_array";

constexpr char CAMERA_INFO_INPUT_COMPONENT_KEY[] = "dope_decoder/camera_model_input";
constexpr char CAMERA_INFO_INPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_camera_info";
constexpr char CAMERA_INFO_INPUT_TOPIC_NAME[] = "camera_info";

constexpr char OUTPUT_COMPONENT_KEY[] = "sink/sink";
constexpr char OUTPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_detection3_d_array";
constexpr char OUTPUT_TOPIC_NAME[] = "dope/detections";

constexpr char APP_YAML_FILENAME[] = "config/dope_node.yaml";
constexpr char PACKAGE_NAME[] = "isaac_ros_dope";

constexpr int kExpectedPoseAsTensorSize = (3 + 4);

const std::vector<std::pair<std::string, std::string>> EXTENSIONS = {
  {"isaac_ros_gxf", "gxf/lib/std/libgxf_std.so"},
  {"isaac_ros_gxf", "gxf/lib/multimedia/libgxf_multimedia.so"},
  {"isaac_ros_gxf", "gxf/lib/serialization/libgxf_serialization.so"},
  {"isaac_ros_gxf", "gxf/lib/cuda/libgxf_cuda.so"},
  {"gxf_isaac_dope", "gxf/lib/libgxf_isaac_dope.so"},
};

const std::vector<std::string> PRESET_EXTENSION_SPEC_NAMES = {};
const std::vector<std::string> EXTENSION_SPEC_FILENAMES = {"config/isaac_ros_dope.yaml"};
const std::vector<std::string> GENERATOR_RULE_FILENAMES = {};
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
const nitros::NitrosPublisherSubscriberConfigMap CONFIG_MAP = {
  {INPUT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(1),
      .compatible_data_format = INPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = INPUT_TOPIC_NAME,
    }
  },
  {CAMERA_INFO_INPUT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(1),
      .compatible_data_format = CAMERA_INFO_INPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = CAMERA_INFO_INPUT_TOPIC_NAME,
    }
  },
  {OUTPUT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(1),
      .compatible_data_format = OUTPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = OUTPUT_TOPIC_NAME,
      .frame_id_source_key = INPUT_COMPONENT_KEY,
    }
  }
};
#pragma GCC diagnostic pop

DopeDecoderNode::DopeDecoderNode(rclcpp::NodeOptions options)
:  nitros::NitrosNode(options,
    APP_YAML_FILENAME,
    CONFIG_MAP,
    PRESET_EXTENSION_SPEC_NAMES,
    EXTENSION_SPEC_FILENAMES,
    GENERATOR_RULE_FILENAMES,
    EXTENSIONS,
    PACKAGE_NAME),
  // Parameters
  configuration_file_(declare_parameter<std::string>("configuration_file", "dope_config.yaml")),
  object_name_(declare_parameter<std::string>("object_name", "Ketchup")),
  tf_frame_name_(declare_parameter<std::string>("tf_frame_name", "dope_object")),
  enable_tf_publishing_(declare_parameter<bool>("enable_tf_publishing", true)),
  map_peak_threshold_(declare_parameter<double>("map_peak_threshold", 0.1)),
  affinity_map_angle_threshold_(declare_parameter<double>("affinity_map_angle_threshold", 0.5)),
  rotation_y_axis_(declare_parameter<double>("rotation_y_axis", false)),
  rotation_x_axis_(declare_parameter<double>("rotation_x_axis", false)),
  rotation_z_axis_(declare_parameter<double>("rotation_z_axis", false)),
  object_dimensions_{},
  camera_matrix_{}
{
  RCLCPP_DEBUG(get_logger(), "[DopeDecoderNode] Constructor");

  // Add callback function for Dope Pose Array to broadcast to ROS TF tree if setting is enabled.
  if (enable_tf_publishing_) {
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    config_map_[OUTPUT_COMPONENT_KEY].callback = std::bind(
      &DopeDecoderNode::DopeDecoderDetectionCallback, this, std::placeholders::_1,
      std::placeholders::_2);
  }

  // Open configuration YAML file
  const std::string package_directory = ament_index_cpp::get_package_share_directory(
    "isaac_ros_dope");
  fs::path yaml_path = package_directory / fs::path("config") / fs::path(configuration_file_);
  if (!fs::exists(yaml_path)) {
    RCLCPP_ERROR(this->get_logger(), "%s could not be found. Exiting.", yaml_path.string().c_str());
    throw std::runtime_error("Parameter parsing failure.");
  }

  // Parse parameters
  rcl_params_t * dope_params = rcl_yaml_node_struct_init(rcutils_get_default_allocator());
  rcl_parse_yaml_file(yaml_path.c_str(), dope_params);

  const std::string dimensions_param = "dimensions." + object_name_;
  rcl_variant_t * dimensions =
    rcl_yaml_node_struct_get("dope", dimensions_param.c_str(), dope_params);
  if (!dimensions->double_array_value) {
    RCLCPP_ERROR(
      this->get_logger(), "No dimensions parameter found for object name: %s",
      object_name_.c_str());
    throw std::runtime_error("Parameter parsing failure.");
  }

  auto dd = dimensions->double_array_value->values;
  object_dimensions_ = {dd[0], dd[1], dd[2]};

  registerSupportedType<nvidia::isaac_ros::nitros::NitrosDetection3DArray>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosCameraInfo>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosTensorList>();

  startNitrosNode();
}

DopeDecoderNode::~DopeDecoderNode() = default;

void DopeDecoderNode::postLoadGraphCallback()
{
  RCLCPP_DEBUG(get_logger(), "[DopeDecoderNode] postLoadGraphCallback().");
  getNitrosContext().setParameter1DFloat64Vector(
    "dope_decoder", "nvidia::isaac_ros::dope::DopeDecoder", "object_dimensions",
    object_dimensions_);

  getNitrosContext().setParameter1DFloat64Vector(
    "dope_decoder", "nvidia::isaac_ros::dope::DopeDecoder", "camera_matrix",
    camera_matrix_);

  getNitrosContext().setParameterStr(
    "dope_decoder", "nvidia::isaac_ros::dope::DopeDecoder", "object_name",
    object_name_);
  getNitrosContext().setParameterFloat64(
    "dope_decoder", "nvidia::isaac_ros::dope::DopeDecoder", "map_peak_threshold",
    map_peak_threshold_);
  getNitrosContext().setParameterFloat64(
    "dope_decoder", "nvidia::isaac_ros::dope::DopeDecoder", "affinity_map_angle_threshold",
    affinity_map_angle_threshold_);
  getNitrosContext().setParameterFloat64(
    "dope_decoder", "nvidia::isaac_ros::dope::DopeDecoder", "rotation_y_axis",
    rotation_y_axis_);
  getNitrosContext().setParameterFloat64(
    "dope_decoder", "nvidia::isaac_ros::dope::DopeDecoder", "rotation_x_axis",
    rotation_x_axis_);
  getNitrosContext().setParameterFloat64(
    "dope_decoder", "nvidia::isaac_ros::dope::DopeDecoder", "rotation_z_axis",
    rotation_z_axis_);
}

// convert Detection3DArray to ROS message that will be published to the TF tree.

void DopeDecoderNode::DopeDecoderDetectionCallback(
  const gxf_context_t context, nitros::NitrosTypeBase & msg)
{
  geometry_msgs::msg::TransformStamped transform_stamped;

  auto msg_entity = nvidia::gxf::Entity::Shared(context, msg.handle);

  // Populate timestamp information back into the header
  auto maybe_input_timestamp = msg_entity->get<nvidia::gxf::Timestamp>();
  if (!maybe_input_timestamp) {    // Fallback to label 'timestamp'
    maybe_input_timestamp = msg_entity->get<nvidia::gxf::Timestamp>("timestamp");
  }
  if (maybe_input_timestamp) {
    transform_stamped.header.stamp.sec = static_cast<int32_t>(
      maybe_input_timestamp.value()->acqtime / static_cast<uint64_t>(1e9));
    transform_stamped.header.stamp.nanosec = static_cast<uint32_t>(
      maybe_input_timestamp.value()->acqtime % static_cast<uint64_t>(1e9));
  } else {
    RCLCPP_WARN(
      get_logger(),
      "[DopeDecoderNode] Failed to get timestamp");
  }

  //  Extract foundation pose list to a struct type defined in detection3_d_array_message.hpp
  auto dope_detections_array_expected = nvidia::isaac::GetDetection3DListMessage(
    msg_entity.value());
  if (!dope_detections_array_expected) {
    RCLCPP_ERROR(
      get_logger(), "[DopeDecoderNode] Failed to get detections data from message entity");
    return;
  }
  auto dope_detections_array = dope_detections_array_expected.value();

  // Extract number of tags detected
  size_t tags_count = dope_detections_array.count;

  /* for each pose instance of a single object (for each Tensor), ennumerate child_frame_id
    in case there are multiple detections in Detection3DArray message (msg_entity)
  */

  int child_frame_id_num = 1;
  if (tags_count > 0) {
    for (size_t i = 0; i < tags_count; i++) {
      auto pose = dope_detections_array.poses.at(i).value();

      transform_stamped.header.frame_id = msg.frame_id;
      transform_stamped.child_frame_id = tf_frame_name_ + std::to_string(child_frame_id_num);
      transform_stamped.transform.translation.x = pose->translation.x();
      transform_stamped.transform.translation.y = pose->translation.y();
      transform_stamped.transform.translation.z = pose->translation.z();
      transform_stamped.transform.rotation.x = pose->rotation.quaternion().x();
      transform_stamped.transform.rotation.y = pose->rotation.quaternion().y();
      transform_stamped.transform.rotation.z = pose->rotation.quaternion().z();
      transform_stamped.transform.rotation.w = pose->rotation.quaternion().w();

      tf_broadcaster_->sendTransform(transform_stamped);
      child_frame_id_num++;
    }
  }
}


}  // namespace dope
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dope::DopeDecoderNode)
