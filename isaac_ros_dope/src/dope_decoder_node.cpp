// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_nitros_pose_array_type/nitros_pose_array.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "rclcpp/rclcpp.hpp"


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

constexpr char OUTPUT_COMPONENT_KEY[] = "vault/vault";
constexpr char OUTPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_pose_array";
constexpr char OUTPUT_TOPIC_NAME[] = "dope/pose_array";

constexpr char APP_YAML_FILENAME[] = "config/dope_node.yaml";
constexpr char PACKAGE_NAME[] = "isaac_ros_dope";

const std::vector<std::pair<std::string, std::string>> EXTENSIONS = {
  {"isaac_ros_gxf", "gxf/lib/std/libgxf_std.so"},
  {"isaac_ros_gxf", "gxf/lib/multimedia/libgxf_multimedia.so"},
  {"isaac_ros_gxf", "gxf/lib/serialization/libgxf_serialization.so"},
  {"isaac_ros_gxf", "gxf/lib/cuda/libgxf_cuda.so"},
  {"isaac_ros_dope", "gxf/lib/dope/libgxf_dope.so"},
};

const std::vector<std::string> PRESET_EXTENSION_SPEC_NAMES = {
  "isaac_ros_dope",
};
const std::vector<std::string> EXTENSION_SPEC_FILENAMES = {};
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
  object_dimensions_{},
  camera_matrix_{}
{
  RCLCPP_DEBUG(get_logger(), "[DopeDecoderNode] Constructor");

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

  rcl_variant_t * cam_mat = rcl_yaml_node_struct_get("dope", "camera_matrix", dope_params);
  if (!cam_mat->double_array_value) {
    RCLCPP_ERROR(this->get_logger(), "No camera_matrix parameter found");
    throw std::runtime_error("Parameter parsing failure.");
  }

  auto vv = cam_mat->double_array_value->values;
  camera_matrix_ = {vv[0], vv[1], vv[2], vv[3], vv[4], vv[5], vv[6], vv[7], vv[8]};

  rcl_yaml_node_struct_fini(dope_params);

  registerSupportedType<nvidia::isaac_ros::nitros::NitrosPoseArray>();
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
}

}  // namespace dope
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dope::DopeDecoderNode)
