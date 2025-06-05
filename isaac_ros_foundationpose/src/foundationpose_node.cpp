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

#include <filesystem>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "detection3_d_array_message/detection3_d_array_message.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/transform_broadcaster.h"

#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_nitros_camera_info_type/nitros_camera_info.hpp"
#include "isaac_ros_nitros_detection3_d_array_type/nitros_detection3_d_array.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"

#include "isaac_ros_foundationpose/foundationpose_node.hpp"

// Helper function to convert StringList to readable string format
std::string StringListToString(const StringList & string_list)
{
  std::string result = "[";
  for (size_t i = 0; i < string_list.size(); ++i) {
    if (i > 0) {result += ", ";}
    result += "\"" + string_list[i] + "\"";
  }
  result += "]";
  return result;
}

namespace nvidia
{
namespace isaac_ros
{
namespace foundationpose
{

using nvidia::gxf::optimizer::GraphIOGroupSupportedDataTypesInfoList;

constexpr char INPUT_DEPTH_COMPONENT_KEY[] = "sync/depth_receiver";
constexpr char INPUT_DEPTH_TENSOR_FORMAT[] = "nitros_image_32FC1";
constexpr char INPUT_DEPTH_TOPIC_NAME[] = "pose_estimation/depth_image";

constexpr char INPUT_RGB_IMAGE_COMPONENT_KEY[] = "sync/rgb_image_receiver";
constexpr char INPUT_RGB_IMAGE_TENSOR_FORMAT[] = "nitros_image_rgb8";
constexpr char INPUT_RGB_IMAGE_TOPIC_NAME[] = "pose_estimation/image";

constexpr char INPUT_CAMERA_INFO_COMPONENT_KEY[] = "sync/camera_model_receiver";
constexpr char INPUT_CAMERA_INFO_FORMAT[] = "nitros_camera_info";
constexpr char INPUT_CAMERA_INFO_TOPIC_NAME[] = "pose_estimation/camera_info";

constexpr char INPUT_SEGMENTATION_COMPONENT_KEY[] = "sync/mask_receiver";
constexpr char INPUT_SEGMENTATION_FORMAT[] = "nitros_image_mono8";
constexpr char INPUT_SEGMENTATION_TOPIC_NAME[] = "pose_estimation/segmentation";

constexpr char OUTPUT_MATRIX_COMPONENT_KEY[] = "pose_matrix_sink/sink";
constexpr char OUTPUT_MATRIX_FORMAT[] = "nitros_tensor_list_nchw";
constexpr char OUTPUT_MATRIX_TOPIC_NAME[] = "pose_estimation/pose_matrix_output";

constexpr char OUTPUT_POSE_COMPONENT_KEY[] = "pose_sink/sink";
constexpr char OUTPUT_POSE_FORMAT[] = "nitros_detection3_d_array";
constexpr char OUTPUT_POSE_TOPIC_NAME[] = "pose_estimation/output";

constexpr char APP_YAML_FILENAME[] = "config/nitros_foundationpose_node.yaml";
constexpr char PACKAGE_NAME[] = "isaac_ros_foundationpose";

const std::vector<std::pair<std::string, std::string>> EXTENSIONS = {
  {"isaac_ros_gxf", "gxf/lib/std/libgxf_std.so"},
  {"isaac_ros_gxf", "gxf/lib/multimedia/libgxf_multimedia.so"},
  {"isaac_ros_gxf", "gxf/lib/serialization/libgxf_serialization.so"},
  {"isaac_ros_gxf", "gxf/lib/cuda/libgxf_cuda.so"},
  {"gxf_isaac_depth_image_proc", "gxf/lib/libgxf_isaac_depth_image_proc.so"},
  {"gxf_isaac_sgm", "gxf/lib/libgxf_isaac_sgm.so"},
  {"gxf_isaac_messages", "gxf/lib/libgxf_isaac_messages.so"},
  {"gxf_isaac_ros_messages", "gxf/lib/libgxf_isaac_ros_messages.so"},
  {"gxf_isaac_tensor_rt", "gxf/lib/libgxf_isaac_tensor_rt.so"},
  {"gxf_isaac_foundationpose", "gxf/lib/libgxf_isaac_foundationpose.so"},
};

const std::vector<std::string> PRESET_EXTENSION_SPEC_NAMES = {};
const std::vector<std::string> EXTENSION_SPEC_FILENAMES = {
  "config/isaac_ros_foundationpose_spec.yaml"
};
const std::vector<std::string> GENERATOR_RULE_FILENAMES = {"config/namespace_injector_rule.yaml"};
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
const nitros::NitrosPublisherSubscriberConfigMap CONFIG_MAP = {
  {INPUT_DEPTH_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = INPUT_DEPTH_TENSOR_FORMAT,
      .topic_name = INPUT_DEPTH_TOPIC_NAME,
    }},
  {INPUT_RGB_IMAGE_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = INPUT_RGB_IMAGE_TENSOR_FORMAT,
      .topic_name = INPUT_RGB_IMAGE_TOPIC_NAME,
    }},
  {INPUT_CAMERA_INFO_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = INPUT_CAMERA_INFO_FORMAT,
      .topic_name = INPUT_CAMERA_INFO_TOPIC_NAME,
    }},
  {INPUT_SEGMENTATION_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = INPUT_SEGMENTATION_FORMAT,
      .topic_name = INPUT_SEGMENTATION_TOPIC_NAME,
    }},
  {OUTPUT_POSE_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = OUTPUT_POSE_FORMAT,
      .topic_name = OUTPUT_POSE_TOPIC_NAME,
      .frame_id_source_key = INPUT_RGB_IMAGE_COMPONENT_KEY,
    }},
  {OUTPUT_MATRIX_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = OUTPUT_MATRIX_FORMAT,
      .topic_name = OUTPUT_MATRIX_TOPIC_NAME,
      .frame_id_source_key = INPUT_RGB_IMAGE_COMPONENT_KEY,
    }},
};
#pragma GCC diagnostic pop

FoundationPoseNode::FoundationPoseNode(rclcpp::NodeOptions options)
: nitros::NitrosNode(
    options, APP_YAML_FILENAME, CONFIG_MAP, PRESET_EXTENSION_SPEC_NAMES, EXTENSION_SPEC_FILENAMES,
    GENERATOR_RULE_FILENAMES, EXTENSIONS, PACKAGE_NAME),
  configuration_file_(
    declare_parameter<std::string>("configuration_file", "foundationpose_model_config.yaml")),
  mesh_file_path_(declare_parameter<std::string>("mesh_file_path", "textured_simple.obj")),
  min_depth_(declare_parameter<float>("min_depth", 0.1)),
  max_depth_(declare_parameter<float>("max_depth", 4.0)),
  refine_iterations_(declare_parameter<int>("refine_iterations", 1)),
  symmetry_axes_(
    declare_parameter<StringList>("symmetry_axes", StringList())),
  symmetry_planes_(
    declare_parameter<StringList>("symmetry_planes", StringList())),
  fixed_axis_angles_(
    declare_parameter<StringList>("fixed_axis_angles", StringList())),
  fixed_translations_(
    declare_parameter<StringList>("fixed_translations", StringList())),

  refine_model_file_path_(
    declare_parameter<std::string>("refine_model_file_path", "/tmp/refine_model.onnx")),
  refine_engine_file_path_(
    declare_parameter<std::string>("refine_engine_file_path", "/tmp/refine_trt_engine.plan")),
  score_model_file_path_(
    declare_parameter<std::string>("score_model_file_path", "/tmp/score_model.onnx")),
  score_engine_file_path_(
    declare_parameter<std::string>("score_engine_file_path", "/tmp/score_trt_engine.plan")),

  refine_input_tensor_names_(
    declare_parameter<StringList>("refine_input_tensor_names", StringList())),
  refine_input_binding_names_(
    declare_parameter<StringList>("refine_input_binding_names", StringList())),
  score_input_tensor_names_(
    declare_parameter<StringList>("score_input_tensor_names", StringList())),
  score_input_binding_names_(
    declare_parameter<StringList>("score_input_binding_names", StringList())),

  refine_output_tensor_names_(
    declare_parameter<StringList>("refine_output_tensor_names", StringList())),
  refine_output_binding_names_(
    declare_parameter<StringList>("refine_output_binding_names", StringList())),
  score_output_tensor_names_(
    declare_parameter<StringList>("score_output_tensor_names", StringList())),
  score_output_binding_names_(
    declare_parameter<StringList>("score_output_binding_names", StringList())),

  discard_time_ms_(declare_parameter<int>("discard_msg_older_than_ms", 1000)),
  discard_old_messages_(declare_parameter<bool>("discard_old_messages", false)),
  pose_estimation_timeout_ms_(declare_parameter<int>("pose_estimation_timeout_ms", 5000)),
  sync_threshold_(declare_parameter<int>("sync_threshold", 0)),
  tf_frame_name_(declare_parameter<std::string>("tf_frame_name", "fp_object")),
  debug_(declare_parameter<bool>("debug", false)),
  debug_dir_(declare_parameter<std::string>("debug_dir", "/tmp/foundationpose"))
{
  RCLCPP_DEBUG(get_logger(), "[FoundationPoseNode] Constructor");

  // This function sets the QoS parameter for publishers and subscribers setup by this NITROS node
  rclcpp::QoS depth_qos_ = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT",
    "depth_qos");
  rclcpp::QoS color_qos_ = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT",
    "color_qos");
  rclcpp::QoS color_info_qos_ = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT",
    "color_info_qos");
  rclcpp::QoS segmentation_qos_ = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT",
    "segmentation_qos");
  for (auto & config : config_map_) {
    if (config.second.topic_name == INPUT_DEPTH_TOPIC_NAME) {
      config.second.qos = depth_qos_;
    }
    if (config.second.topic_name == INPUT_RGB_IMAGE_TOPIC_NAME) {
      config.second.qos = color_qos_;
    }
    if (config.second.topic_name == INPUT_CAMERA_INFO_TOPIC_NAME) {
      config.second.qos = color_info_qos_;
    }
    if (config.second.topic_name == INPUT_SEGMENTATION_TOPIC_NAME) {
      config.second.qos = segmentation_qos_;
    }
  }

  // Add callback function for FoundationPose Detection3D array to broadcast to ROS TF tree
  config_map_[OUTPUT_POSE_COMPONENT_KEY].callback = std::bind(
    &FoundationPoseNode::FoundationPoseDetectionCallback, this, std::placeholders::_1,
    std::placeholders::_2);

  // Open configuration YAML file
  const std::string package_directory = ament_index_cpp::get_package_share_directory(
    "isaac_ros_foundationpose");
  std::filesystem::path yaml_path =
    package_directory / std::filesystem::path("config") /
    std::filesystem::path(configuration_file_);
  if (!std::filesystem::exists(yaml_path)) {
    RCLCPP_ERROR(this->get_logger(), "%s could not be found. Exiting.", yaml_path.string().c_str());
    throw std::runtime_error("Parameter parsing failure.");
  }

  // Parse parameters
  rcl_params_t * foundationpose_params = rcl_yaml_node_struct_init(rcutils_get_default_allocator());
  rcl_parse_yaml_file(yaml_path.c_str(), foundationpose_params);

  rcl_variant_t * max_hypothesis = rcl_yaml_node_struct_get(
    "foundationpose", "max_hypothesis", foundationpose_params);
  if (!max_hypothesis->integer_value) {
    RCLCPP_ERROR(this->get_logger(), "No max_hypothesis parameter found");
    throw std::runtime_error("Parameter parsing failure.");
  }
  max_hypothesis_ = static_cast<uint32_t>(*max_hypothesis->integer_value);

  rcl_variant_t * resized_image_width = rcl_yaml_node_struct_get(
    "foundationpose", "resized_image_width", foundationpose_params);
  if (!resized_image_width->integer_value) {
    RCLCPP_ERROR(this->get_logger(), "No resized_image_width parameter found");
    throw std::runtime_error("Parameter parsing failure.");
  }
  resized_image_width_ = static_cast<uint32_t>(*resized_image_width->integer_value);

  rcl_variant_t * resized_image_height = rcl_yaml_node_struct_get(
    "foundationpose", "resized_image_height", foundationpose_params);
  if (!resized_image_width->integer_value) {
    RCLCPP_ERROR(this->get_logger(), "No resized_image_height parameter found");
    throw std::runtime_error("Parameter parsing failure.");
  }
  resized_image_height_ = static_cast<uint32_t>(*resized_image_height->integer_value);

  rcl_variant_t * refine_crop_ratio = rcl_yaml_node_struct_get(
    "foundationpose", "refine_crop_ratio", foundationpose_params);
  if (!refine_crop_ratio->double_value) {
    RCLCPP_ERROR(this->get_logger(), "No refine_crop_ratio parameter found");
    throw std::runtime_error("Parameter parsing failure.");
  }
  refine_crop_ratio_ = static_cast<float>(*refine_crop_ratio->double_value);

  rcl_variant_t * score_crop_ratio = rcl_yaml_node_struct_get(
    "foundationpose", "score_crop_ratio", foundationpose_params);
  if (!score_crop_ratio->double_value) {
    RCLCPP_ERROR(this->get_logger(), "No score_crop_ratio parameter found");
    throw std::runtime_error("Parameter parsing failure.");
  }
  score_crop_ratio_ = static_cast<float>(*score_crop_ratio->double_value);

  rcl_variant_t * rot_normalizer = rcl_yaml_node_struct_get(
    "foundationpose", "rot_normalizer", foundationpose_params);
  if (!rot_normalizer->double_value) {
    RCLCPP_ERROR(this->get_logger(), "No rot_normalizer parameter found");
    throw std::runtime_error("Parameter parsing failure.");
  }
  rot_normalizer_ = static_cast<float>(*rot_normalizer->double_value);

  rcl_yaml_node_struct_fini(foundationpose_params);

  param_subscriber_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
  mesh_file_path_param_cb_handle_ = param_subscriber_->add_parameter_callback(
    "mesh_file_path",
    std::bind(&FoundationPoseNode::UpdateMeshFilePathCallback, this, std::placeholders::_1));
  tf_frame_name_param_cb_handle_ = param_subscriber_->add_parameter_callback(
    "tf_frame_name",
    std::bind(&FoundationPoseNode::UpdateTfFrameNameCallback, this, std::placeholders::_1));

  // Register parameter callbacks for constraint parameters
  fixed_translations_param_cb_handle_ = param_subscriber_->add_parameter_callback(
    "fixed_translations",
    std::bind(&FoundationPoseNode::UpdateFixedTranslationsCallback, this, std::placeholders::_1));

  fixed_axis_angles_param_cb_handle_ = param_subscriber_->add_parameter_callback(
    "fixed_axis_angles",
    std::bind(&FoundationPoseNode::UpdateFixedAxisAnglesCallback, this, std::placeholders::_1));

  symmetry_axes_param_cb_handle_ = param_subscriber_->add_parameter_callback(
    "symmetry_axes",
    std::bind(&FoundationPoseNode::UpdateSymmetryAxesCallback, this, std::placeholders::_1));

  symmetry_planes_param_cb_handle_ = param_subscriber_->add_parameter_callback(
    "symmetry_planes",
    std::bind(&FoundationPoseNode::UpdateSymmetryPlanesCallback, this, std::placeholders::_1));

  registerSupportedType<nvidia::isaac_ros::nitros::NitrosCameraInfo>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosImage>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosDetection3DArray>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosTensorList>();

  startNitrosNode();
}

FoundationPoseNode::~FoundationPoseNode() = default;

void FoundationPoseNode::postLoadGraphCallback()
{
  RCLCPP_DEBUG(get_logger(), "[FoundationPoseNode] postLoadGraphCallback().");

  // Set the model configuration file path from parameter
  getNitrosContext().setParameterUInt32(
    "sampling", "nvidia::isaac_ros::FoundationposeSampling", "max_hypothesis",
    max_hypothesis_);

  getNitrosContext().setParameterUInt32(
    "render", "nvidia::isaac_ros::FoundationposeRender", "resized_image_width",
    resized_image_width_);

  getNitrosContext().setParameterUInt32(
    "render_score", "nvidia::isaac_ros::FoundationposeRender", "resized_image_width",
    resized_image_width_);

  getNitrosContext().setParameterUInt32(
    "render", "nvidia::isaac_ros::FoundationposeRender", "resized_image_height",
    resized_image_height_);

  getNitrosContext().setParameterUInt32(
    "render_score", "nvidia::isaac_ros::FoundationposeRender", "resized_image_height",
    resized_image_height_);

  getNitrosContext().setParameterFloat32(
    "render", "nvidia::isaac_ros::FoundationposeRender", "crop_ratio", refine_crop_ratio_);

  getNitrosContext().setParameterFloat32(
    "render_score", "nvidia::isaac_ros::FoundationposeRender", "crop_ratio", score_crop_ratio_);

  getNitrosContext().setParameterFloat32(
    "transform", "nvidia::isaac_ros::FoundationposeTransformation", "rot_normalizer",
    rot_normalizer_);

  // Set the mesh path from parameter
  getNitrosContext().setParameterStr(
    "utils", "nvidia::isaac_ros::MeshStorage", "mesh_file_path", mesh_file_path_);

  // Set the depth threshold
  getNitrosContext().setParameterFloat32(
    "render", "nvidia::isaac_ros::FoundationposeRender", "min_depth", min_depth_);

  getNitrosContext().setParameterFloat32(
    "render_score", "nvidia::isaac_ros::FoundationposeRender", "min_depth", min_depth_);

  getNitrosContext().setParameterFloat32(
    "render", "nvidia::isaac_ros::FoundationposeRender", "max_depth", max_depth_);

  getNitrosContext().setParameterFloat32(
    "render_score", "nvidia::isaac_ros::FoundationposeRender", "max_depth", max_depth_);

  getNitrosContext().setParameterFloat32(
    "sampling", "nvidia::isaac_ros::FoundationposeSampling", "min_depth", min_depth_);

  // Set the refine iterations
  getNitrosContext().setParameterInt32(
    "render", "nvidia::isaac_ros::FoundationposeRender", "refine_iterations",
    refine_iterations_);
  getNitrosContext().setParameterInt32(
    "transform", "nvidia::isaac_ros::FoundationposeTransformation", "refine_iterations",
    refine_iterations_);

  // Set the refine network TensorRT configs from parameter
  getNitrosContext().setParameterStr(
    "refine_inference", "nvidia::gxf::TensorRtInference", "model_file_path",
    refine_model_file_path_);

  getNitrosContext().setParameterStr(
    "refine_inference", "nvidia::gxf::TensorRtInference", "engine_file_path",
    refine_engine_file_path_);

  getNitrosContext().setParameter1DStrVector(
    "refine_inference", "nvidia::gxf::TensorRtInference", "input_tensor_names",
    refine_input_tensor_names_);

  getNitrosContext().setParameter1DStrVector(
    "refine_inference", "nvidia::gxf::TensorRtInference", "input_binding_names",
    refine_input_binding_names_);

  getNitrosContext().setParameter1DStrVector(
    "refine_inference", "nvidia::gxf::TensorRtInference", "output_tensor_names",
    refine_output_tensor_names_);

  getNitrosContext().setParameter1DStrVector(
    "refine_inference", "nvidia::gxf::TensorRtInference", "output_binding_names",
    refine_output_binding_names_);


  // Set the score network TensorRT configs from parameter
  getNitrosContext().setParameterStr(
    "score_inference", "nvidia::gxf::TensorRtInference", "model_file_path",
    score_model_file_path_);

  getNitrosContext().setParameterStr(
    "score_inference", "nvidia::gxf::TensorRtInference", "engine_file_path",
    score_engine_file_path_);

  getNitrosContext().setParameter1DStrVector(
    "score_inference", "nvidia::gxf::TensorRtInference", "input_tensor_names",
    score_input_tensor_names_);

  getNitrosContext().setParameter1DStrVector(
    "score_inference", "nvidia::gxf::TensorRtInference", "input_binding_names",
    score_input_binding_names_);

  getNitrosContext().setParameter1DStrVector(
    "score_inference", "nvidia::gxf::TensorRtInference", "output_tensor_names",
    score_output_tensor_names_);

  getNitrosContext().setParameter1DStrVector(
    "score_inference", "nvidia::gxf::TensorRtInference", "output_binding_names",
    score_output_binding_names_);

  getNitrosContext().setParameterBool(
    "sync", "nvidia::isaac_ros::FoundationPoseSynchronization", "discard_old_messages",
    discard_old_messages_
  );
  getNitrosContext().setParameterInt64(
    "sync", "nvidia::isaac_ros::FoundationPoseSynchronization", "discard_time_ms",
    discard_time_ms_
  );
  getNitrosContext().setParameterInt64(
    "sync", "nvidia::isaac_ros::FoundationPoseSynchronization", "pose_estimation_timeout_ms",
    pose_estimation_timeout_ms_
  );
  getNitrosContext().setParameterInt64(
    "sync", "nvidia::isaac_ros::FoundationPoseSynchronization", "sync_threshold",
    sync_threshold_
  );

  // Set symmetry planes for backward compatibility if any
  if (symmetry_planes_.size() > 0) {
    getNitrosContext().setParameter1DStrVector(
      "sampling", "nvidia::isaac_ros::FoundationposeSampling", "symmetry_planes",
      symmetry_planes_);
  }

  if (symmetry_axes_.size() > 0) {
    getNitrosContext().setParameter1DStrVector(
      "sampling", "nvidia::isaac_ros::FoundationposeSampling", "symmetry_axes",
      symmetry_axes_);
  }

  // Set fixed axis angle constraints if any
  if (fixed_axis_angles_.size() > 0) {
    getNitrosContext().setParameter1DStrVector(
      "sampling", "nvidia::isaac_ros::FoundationposeSampling", "fixed_axis_angles",
      fixed_axis_angles_);
  }

  // Set the fixed translations parameter
  if (fixed_translations_.size() > 0) {
    getNitrosContext().setParameter1DStrVector(
      "sampling", "nvidia::isaac_ros::FoundationposeSampling", "fixed_translations",
      fixed_translations_);
  }

  // Set debug mode
  getNitrosContext().setParameterBool(
    "render", "nvidia::isaac_ros::FoundationposeRender", "debug",
    debug_);
  getNitrosContext().setParameterBool(
    "render_score", "nvidia::isaac_ros::FoundationposeRender", "debug",
    debug_);
  getNitrosContext().setParameterStr(
    "render", "nvidia::isaac_ros::FoundationposeRender", "debug_dir",
    debug_dir_);
  getNitrosContext().setParameterStr(
    "render_score", "nvidia::isaac_ros::FoundationposeRender", "debug_dir",
    debug_dir_);
}

void FoundationPoseNode::UpdateTfFrameNameCallback(const rclcpp::Parameter & tf_frame_name)
{
  std::unique_lock<std::mutex> lock(mutex_);
  tf_frame_name_ = tf_frame_name.as_string();
  RCLCPP_INFO(
    get_logger(),
    "[FoundationPoseNode] Changing tf frame name to %s",
    tf_frame_name_.c_str());
}

void FoundationPoseNode::UpdateMeshFilePathCallback(const rclcpp::Parameter & mesh_file_path)
{
  mesh_file_path_ = mesh_file_path.as_string();
  RCLCPP_INFO(
    get_logger(),
    "[FoundationPoseNode] Changing mesh file path to %s",
    mesh_file_path_.c_str());

  getNitrosContext().setParameterStr(
    "utils", "nvidia::isaac_ros::MeshStorage", "mesh_file_path", mesh_file_path_);
}

void FoundationPoseNode::UpdateFixedTranslationsCallback(const rclcpp::Parameter & param)
{
  fixed_translations_ = param.as_string_array();
  RCLCPP_INFO(
    get_logger(),
    "[FoundationPoseNode] Changing fixed_translations to %s",
    StringListToString(fixed_translations_).c_str());

  getNitrosContext().setParameter1DStrVector(
    "sampling", "nvidia::isaac_ros::FoundationposeSampling", "fixed_translations",
    fixed_translations_);
}

void FoundationPoseNode::UpdateFixedAxisAnglesCallback(const rclcpp::Parameter & param)
{
  fixed_axis_angles_ = param.as_string_array();
  RCLCPP_INFO(
    get_logger(),
    "[FoundationPoseNode] Changing fixed_axis_angles to %s",
    StringListToString(fixed_axis_angles_).c_str());

  getNitrosContext().setParameter1DStrVector(
    "sampling", "nvidia::isaac_ros::FoundationposeSampling", "fixed_axis_angles",
    fixed_axis_angles_);
}

void FoundationPoseNode::UpdateSymmetryAxesCallback(const rclcpp::Parameter & param)
{
  symmetry_axes_ = param.as_string_array();
  RCLCPP_INFO(
    get_logger(),
    "[FoundationPoseNode] Changing symmetry_axes to %s",
    StringListToString(symmetry_axes_).c_str());

  getNitrosContext().setParameter1DStrVector(
    "sampling", "nvidia::isaac_ros::FoundationposeSampling", "symmetry_axes", symmetry_axes_);
}

void FoundationPoseNode::UpdateSymmetryPlanesCallback(const rclcpp::Parameter & param)
{
  symmetry_planes_ = param.as_string_array();
  RCLCPP_INFO(
    get_logger(),
    "[FoundationPoseNode] Changing symmetry_planes to %s",
    StringListToString(symmetry_planes_).c_str());

  getNitrosContext().setParameter1DStrVector(
    "sampling", "nvidia::isaac_ros::FoundationposeSampling", "symmetry_planes", symmetry_planes_);
}

void FoundationPoseNode::FoundationPoseDetectionCallback(
  const gxf_context_t context, nitros::NitrosTypeBase & msg)
{
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_ =
    std::make_unique<tf2_ros::TransformBroadcaster>(*this);
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
      "[FoundationPoseNode] Failed to get timestamp");
  }

  //  Extract foundation pose list to a struct type defined in detection3_d_array_message.hpp
  auto foundation_pose_detections_array_expected = nvidia::isaac::GetDetection3DListMessage(
    msg_entity.value());
  if (!foundation_pose_detections_array_expected) {
    RCLCPP_ERROR(
      get_logger(), "[FoundationPoseNode] Failed to get detections data from message entity");
    return;
  }
  auto foundation_pose_detections_array = foundation_pose_detections_array_expected.value();

  // Extract number of tags detected
  size_t tags_count = foundation_pose_detections_array.count;

  if (tags_count > 0) {
    std::unique_lock<std::mutex> lock(mutex_);

    // struct is defined in fiducial_message.hpp
    auto pose = foundation_pose_detections_array.poses.at(0).value();

    transform_stamped.header.frame_id = msg.frame_id;
    transform_stamped.child_frame_id = tf_frame_name_;
    transform_stamped.transform.translation.x = pose->translation.x();
    transform_stamped.transform.translation.y = pose->translation.y();
    transform_stamped.transform.translation.z = pose->translation.z();
    transform_stamped.transform.rotation.x = pose->rotation.quaternion().x();
    transform_stamped.transform.rotation.y = pose->rotation.quaternion().y();
    transform_stamped.transform.rotation.z = pose->rotation.quaternion().z();
    transform_stamped.transform.rotation.w = pose->rotation.quaternion().w();

    tf_broadcaster_->sendTransform(transform_stamped);
  }
}

}  // namespace foundationpose
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::foundationpose::FoundationPoseNode)
