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

#include "isaac_ros_foundationpose/foundationpose_node.hpp"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <future>
#include <string>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "cuda_buffer/cuda_buffer_api.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"
#include "rcl_yaml_param_parser/parser.h"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "std_msgs/msg/header.hpp"
#include "foundationpose_sampling.cu.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace foundationpose
{

constexpr uint8_t kFloatCode = static_cast<uint8_t>(TrtConv::DLDataTypeCode::kFloat);

FoundationPoseNode::FoundationPoseNode(rclcpp::NodeOptions options)
: rclcpp::Node("foundationpose_node", options),
  configuration_file_(
    declare_parameter<std::string>("configuration_file", "foundationpose_model_config.yaml")),
  mesh_file_path_(declare_parameter<std::string>("mesh_file_path", "")),
  refine_input_tensor_names_(declare_parameter<StringList>(
      "refine_input_tensor_names", StringList{"input_tensor1", "input_tensor2"})),
  score_input_tensor_names_(declare_parameter<StringList>(
      "score_input_tensor_names", StringList{"input_tensor1", "input_tensor2"})),
  min_depth_(declare_parameter<double>("min_depth", 0.1)),
  max_depth_(declare_parameter<double>("max_depth", 2.0)),
  tf_frame_name_(declare_parameter<std::string>("tf_frame_name", "fp_object")),
  pose_estimation_timeout_ms_(declare_parameter<int64_t>("pose_estimation_timeout_ms", 5000)),
  symmetry_axes_(declare_parameter<StringList>("symmetry_axes", StringList{})),
  symmetry_planes_(declare_parameter<StringList>("symmetry_planes", StringList{})),
  fixed_axis_angles_(declare_parameter<StringList>("fixed_axis_angles", StringList{})),
  fixed_translations_(declare_parameter<StringList>("fixed_translations", StringList{}))
{
  const std::string package_directory = ament_index_cpp::get_package_share_directory(
    "isaac_ros_foundationpose");
  std::filesystem::path yaml_path =
    std::filesystem::path(package_directory) / std::filesystem::path("config") /
    std::filesystem::path(configuration_file_);
  if (!std::filesystem::exists(yaml_path)) {
    RCLCPP_ERROR(get_logger(), "%s could not be found. Exiting.", yaml_path.string().c_str());
    throw std::runtime_error("Parameter parsing failure.");
  }

  rcl_params_t * foundationpose_params =
    rcl_yaml_node_struct_init(rcutils_get_default_allocator());
  rcl_parse_yaml_file(yaml_path.c_str(), foundationpose_params);

  rcl_variant_t * max_hypothesis = rcl_yaml_node_struct_get(
    "foundationpose", "max_hypothesis", foundationpose_params);
  if (!max_hypothesis->integer_value) {
    RCLCPP_ERROR(get_logger(), "No max_hypothesis parameter found");
    throw std::runtime_error("Parameter parsing failure.");
  }
  max_hypothesis_ = static_cast<uint32_t>(*max_hypothesis->integer_value);

  rcl_variant_t * resized_image_width = rcl_yaml_node_struct_get(
    "foundationpose", "resized_image_width", foundationpose_params);
  if (!resized_image_width->integer_value) {
    RCLCPP_ERROR(get_logger(), "No resized_image_width parameter found");
    throw std::runtime_error("Parameter parsing failure.");
  }
  resized_image_width_ = static_cast<uint32_t>(*resized_image_width->integer_value);

  rcl_variant_t * resized_image_height = rcl_yaml_node_struct_get(
    "foundationpose", "resized_image_height", foundationpose_params);
  if (!resized_image_height->integer_value) {
    RCLCPP_ERROR(get_logger(), "No resized_image_height parameter found");
    throw std::runtime_error("Parameter parsing failure.");
  }
  resized_image_height_ = static_cast<uint32_t>(*resized_image_height->integer_value);

  rcl_variant_t * refine_crop_ratio = rcl_yaml_node_struct_get(
    "foundationpose", "refine_crop_ratio", foundationpose_params);
  if (!refine_crop_ratio->double_value) {
    RCLCPP_ERROR(get_logger(), "No refine_crop_ratio parameter found");
    throw std::runtime_error("Parameter parsing failure.");
  }
  refine_crop_ratio_ = static_cast<float>(*refine_crop_ratio->double_value);

  rcl_variant_t * score_crop_ratio = rcl_yaml_node_struct_get(
    "foundationpose", "score_crop_ratio", foundationpose_params);
  if (!score_crop_ratio->double_value) {
    RCLCPP_ERROR(get_logger(), "No score_crop_ratio parameter found");
    throw std::runtime_error("Parameter parsing failure.");
  }
  score_crop_ratio_ = static_cast<float>(*score_crop_ratio->double_value);

  rcl_variant_t * rot_normalizer = rcl_yaml_node_struct_get(
    "foundationpose", "rot_normalizer", foundationpose_params);
  if (!rot_normalizer->double_value) {
    RCLCPP_ERROR(get_logger(), "No rot_normalizer parameter found");
    throw std::runtime_error("Parameter parsing failure.");
  }
  rot_normalizer_ = static_cast<float>(*rot_normalizer->double_value);

  rcl_yaml_node_struct_fini(foundationpose_params);

  refine_iterations_ = declare_parameter<int>("refine_iterations", 3);

  initializePipeline();

  rclcpp::QoS output_qos = ::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos");

  rclcpp::QoS color_qos =
    ::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "color_qos", 10);
  rclcpp::QoS depth_qos =
    ::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "depth_qos", 10);
  rclcpp::QoS color_info_qos =
    ::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "color_info_qos", 10);
  rclcpp::QoS segmentation_qos =
    ::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "segmentation_qos", 10);

  rclcpp::SubscriptionOptions image_sub_options;
  image_sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  image_sub_options.acceptable_buffer_backends = "any";

  rgb_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>();
  rgb_sub_->subscribe(this, "pose_estimation/image", color_qos, image_sub_options);
  depth_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>();
  depth_sub_->subscribe(this, "pose_estimation/depth_image", depth_qos, image_sub_options);
  cam_info_sub_ =
    std::make_shared<message_filters::Subscriber<sensor_msgs::msg::CameraInfo>>();
  cam_info_sub_->subscribe(this, "pose_estimation/camera_info", color_info_qos);
  mask_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>();
  mask_sub_->subscribe(this, "pose_estimation/segmentation", segmentation_qos, image_sub_options);

  // Synchronizer window matches the largest configured QoS depth so it can hold
  // enough messages from every channel to find a sync point.
  const size_t sync_depth = std::max(
    {color_qos.depth(), depth_qos.depth(), color_info_qos.depth(),
      segmentation_qos.depth()});
  sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
    SyncPolicy(sync_depth), *rgb_sub_, *depth_sub_, *cam_info_sub_, *mask_sub_);
  sync_->registerCallback(
    std::bind(
      &FoundationPoseNode::syncCallback, this,
      std::placeholders::_1, std::placeholders::_2,
      std::placeholders::_3, std::placeholders::_4));

  // Output publishers
  detection_pub_ = create_publisher<vision_msgs::msg::Detection3DArray>(
    "pose_estimation/output", output_qos);
  reset_client_ = create_client<std_srvs::srv::Trigger>("selector/reset");
  tracking_switch_mesh_client_ =
    create_client<isaac_ros_foundationpose::srv::SwitchMesh>("tracking/switch_mesh");
  rclcpp::PublisherOptions pose_pub_options;
  pose_pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  pose_matrix_pub_ = create_publisher<TensorMsg::TensorListMsg>(
    "pose_estimation/pose_matrix_output", output_qos, pose_pub_options);

  rclcpp::PublisherOptions trt_pub_options;
  trt_pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  refine_pub_ = create_publisher<TensorMsg::TensorListMsg>(
    "refine/tensor_pub", rclcpp::QoS(1), trt_pub_options);
  score_pub_ = create_publisher<TensorMsg::TensorListMsg>(
    "score/tensor_pub", rclcpp::QoS(1), trt_pub_options);

  // Dedicated reentrant callback group for TRT result subscribers so their
  // dispatch is not blocked behind syncCallback / parameter callbacks on the
  // default callback group when the executor is busy.
  trt_callback_group_ = create_callback_group(
    rclcpp::CallbackGroupType::Reentrant);
  rclcpp::SubscriptionOptions trt_sub_options;
  trt_sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  trt_sub_options.acceptable_buffer_backends = "any";
  trt_sub_options.callback_group = trt_callback_group_;

  refine_sub_ = create_subscription<TensorMsg::TensorListMsg>(
    "refine/tensor_sub", rclcpp::QoS(1),
    std::bind(&FoundationPoseNode::onRefineResult, this, std::placeholders::_1),
    trt_sub_options);
  score_sub_ = create_subscription<TensorMsg::TensorListMsg>(
    "score/tensor_sub", rclcpp::QoS(1),
    std::bind(&FoundationPoseNode::onScoreResult, this, std::placeholders::_1),
    trt_sub_options);

  watchdog_timer_ = create_wall_timer(
    std::chrono::milliseconds(pose_estimation_timeout_ms_),
    [this]() {
      if (processing_.load()) {
        RCLCPP_WARN(get_logger(), "[FoundationPoseNode] Processing timeout, releasing gate");
        processing_.store(false);
      }
    });

  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  param_subscriber_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
  mesh_file_path_cb_ = param_subscriber_->add_parameter_callback(
    "mesh_file_path",
    [this](const rclcpp::Parameter & param) {
      const auto new_mesh_file_path = param.as_string();
      if (reset_client_->service_is_ready()) {
        auto future = reset_client_->async_send_request(
          std::make_shared<std_srvs::srv::Trigger::Request>());
        if (future.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready) {
          RCLCPP_WARN(
            get_logger(),
            "[FoundationPoseNode] selector/reset service timed out; continuing mesh switch");
        } else if (!future.get()->success) {
          RCLCPP_WARN(
            get_logger(),
            "[FoundationPoseNode] selector/reset failed; continuing mesh switch");
        }
      } else {
        RCLCPP_DEBUG(
          get_logger(),
          "[FoundationPoseNode] selector/reset service is not ready; "
          "selector may wait for timeout");
      }
      {
        std::lock_guard<std::mutex> lock(mesh_mutex_);
        mesh_file_path_ = new_mesh_file_path;
        mesh_loader_->tryReload(mesh_file_path_);
      }
      if (tracking_switch_mesh_client_->service_is_ready()) {
        auto request = std::make_shared<isaac_ros_foundationpose::srv::SwitchMesh::Request>();
        request->mesh_file_path = new_mesh_file_path;
        auto future = tracking_switch_mesh_client_->async_send_request(request);
        if (future.wait_for(std::chrono::milliseconds(500)) != std::future_status::ready) {
          RCLCPP_WARN(
            get_logger(),
            "[FoundationPoseNode] tracking/switch_mesh service timed out; "
            "tracking mesh may be stale");
        } else {
          auto response = future.get();
          if (!response->success) {
            RCLCPP_WARN(
              get_logger(), "[FoundationPoseNode] tracking/switch_mesh failed: %s",
              response->message.c_str());
          }
        }
      } else {
        RCLCPP_DEBUG(
          get_logger(),
          "[FoundationPoseNode] tracking/switch_mesh service is not ready; "
          "tracking mesh was not updated");
      }
    });
  tf_frame_name_cb_ = param_subscriber_->add_parameter_callback(
    "tf_frame_name",
    [this](const rclcpp::Parameter & param) {
      std::lock_guard<std::mutex> lock(param_mutex_);
      tf_frame_name_ = param.as_string();
    });
  fixed_translations_cb_ = param_subscriber_->add_parameter_callback(
    "fixed_translations",
    [this](const rclcpp::Parameter & param) {
      std::lock_guard<std::mutex> lock(param_mutex_);
      fixed_translations_ = param.as_string_array();
      PoseSamplerParams p = {};
      p.max_hypothesis = max_hypothesis_;
      p.min_depth = min_depth_;
      p.symmetry_axes = symmetry_axes_;
      p.symmetry_planes = symmetry_planes_;
      p.fixed_axis_angles = fixed_axis_angles_;
      p.fixed_translations = fixed_translations_;
      sampler_->updateParams(p);
    });
  fixed_axis_angles_cb_ = param_subscriber_->add_parameter_callback(
    "fixed_axis_angles",
    [this](const rclcpp::Parameter & param) {
      std::lock_guard<std::mutex> lock(param_mutex_);
      fixed_axis_angles_ = param.as_string_array();
      PoseSamplerParams p = {};
      p.max_hypothesis = max_hypothesis_;
      p.min_depth = min_depth_;
      p.symmetry_axes = symmetry_axes_;
      p.symmetry_planes = symmetry_planes_;
      p.fixed_axis_angles = fixed_axis_angles_;
      p.fixed_translations = fixed_translations_;
      sampler_->updateParams(p);
    });
  symmetry_axes_cb_ = param_subscriber_->add_parameter_callback(
    "symmetry_axes",
    [this](const rclcpp::Parameter & param) {
      std::lock_guard<std::mutex> lock(param_mutex_);
      symmetry_axes_ = param.as_string_array();
      PoseSamplerParams p = {};
      p.max_hypothesis = max_hypothesis_;
      p.min_depth = min_depth_;
      p.symmetry_axes = symmetry_axes_;
      p.symmetry_planes = symmetry_planes_;
      p.fixed_axis_angles = fixed_axis_angles_;
      p.fixed_translations = fixed_translations_;
      sampler_->updateParams(p);
    });
  symmetry_planes_cb_ = param_subscriber_->add_parameter_callback(
    "symmetry_planes",
    [this](const rclcpp::Parameter & param) {
      std::lock_guard<std::mutex> lock(param_mutex_);
      symmetry_planes_ = param.as_string_array();
      PoseSamplerParams p = {};
      p.max_hypothesis = max_hypothesis_;
      p.min_depth = min_depth_;
      p.symmetry_axes = symmetry_axes_;
      p.symmetry_planes = symmetry_planes_;
      p.fixed_axis_angles = fixed_axis_angles_;
      p.fixed_translations = fixed_translations_;
      sampler_->updateParams(p);
    });

  RCLCPP_INFO(get_logger(), "[FoundationPoseNode] Initialization complete");
}

FoundationPoseNode::~FoundationPoseNode()
{
  if (processing_thread_.joinable()) {
    processing_thread_.join();
  }
  renderer_.reset();
  score_renderer_.reset();
  sampler_.reset();
  decoder_.reset();
  transformer_.reset();
  mesh_loader_.reset();
  if (pc_gpu_) {cudaFree(pc_gpu_);}
  if (all_poses_gpu_) {cudaFree(all_poses_gpu_);}
  // Pools self-destroy in their dtors.
  if (cuda_stream_) {cudaStreamDestroy(cuda_stream_);}
}

void FoundationPoseNode::initializePipeline()
{
  cudaStreamCreate(&cuda_stream_);

  mesh_loader_ = std::make_unique<MeshLoader>(cuda_stream_);
  if (!mesh_file_path_.empty()) {
    mesh_loader_->load(mesh_file_path_);
  }

  PoseSamplerParams sampler_params;
  sampler_params.max_hypothesis = max_hypothesis_;
  sampler_params.min_depth = min_depth_;
  sampler_params.symmetry_axes = symmetry_axes_;
  sampler_params.symmetry_planes = symmetry_planes_;
  sampler_params.fixed_axis_angles = fixed_axis_angles_;
  sampler_params.fixed_translations = fixed_translations_;
  sampler_ = std::make_unique<PoseSampler>(sampler_params, cuda_stream_);

  PoseRendererParams render_params;
  render_params.crop_ratio = refine_crop_ratio_;
  render_params.min_depth = min_depth_;
  render_params.max_depth = max_depth_;
  render_params.resized_height = resized_image_height_;
  render_params.resized_width = resized_image_width_;
  renderer_ = std::make_unique<PoseRenderer>(render_params, cuda_stream_);

  PoseRendererParams score_render_params = render_params;
  score_render_params.crop_ratio = score_crop_ratio_;
  score_renderer_ = std::make_unique<PoseRenderer>(score_render_params, cuda_stream_);

  transformer_ = std::make_unique<PoseTransformer>(rot_normalizer_, cuda_stream_);
  decoder_ = std::make_unique<PoseDecoder>(cuda_stream_);

  size_t poses_bytes = static_cast<size_t>(max_hypothesis_) * 16 * sizeof(float);
  CHECK_CUDA_ERROR(cudaMalloc(&all_poses_gpu_, poses_bytes), "malloc all_poses_gpu");

  RCLCPP_INFO(get_logger(), "[FoundationPoseNode] Pipeline initialized");
}

// --- Blocking TRT call helpers ---

void FoundationPoseNode::onRefineResult(
  const TensorMsg::TensorListMsg::ConstSharedPtr & result)
{
  std::lock_guard<std::mutex> lock(refine_mutex_);
  refine_result_ = *result;
  refine_result_ready_ = true;
  refine_cv_.notify_one();
}

void FoundationPoseNode::onScoreResult(
  const TensorMsg::TensorListMsg::ConstSharedPtr & result)
{
  std::lock_guard<std::mutex> lock(score_mutex_);
  score_result_ = *result;
  score_result_ready_ = true;
  score_cv_.notify_one();
}

TensorMsg::TensorListMsg FoundationPoseNode::callRefineTRT(
  TensorMsg::TensorListMsg input)
{
  constexpr auto kTimeout = std::chrono::seconds(5);
  std::unique_lock<std::mutex> lock(refine_mutex_);
  refine_result_ready_ = false;
  refine_pub_->publish(std::move(input));
  if (!refine_cv_.wait_for(lock, kTimeout, [this] {return refine_result_ready_;})) {
    RCLCPP_ERROR(get_logger(),
      "Refine TRT timed out after %" PRId64 "s; check that the TensorRT node is running",
      static_cast<int64_t>(kTimeout.count()));
    return TensorMsg::TensorListMsg();
  }
  return std::move(refine_result_);
}

TensorMsg::TensorListMsg FoundationPoseNode::callScoreTRT(
  TensorMsg::TensorListMsg input)
{
  constexpr auto kTimeout = std::chrono::seconds(5);
  std::unique_lock<std::mutex> lock(score_mutex_);
  score_result_ready_ = false;
  score_pub_->publish(std::move(input));
  if (!score_cv_.wait_for(lock, kTimeout, [this] {return score_result_ready_;})) {
    RCLCPP_ERROR(get_logger(),
      "Score TRT timed out after %" PRId64 "s; check that the TensorRT node is running",
      static_cast<int64_t>(kTimeout.count()));
    return TensorMsg::TensorListMsg();
  }
  return std::move(score_result_);
}

// --- Callbacks ---

void FoundationPoseNode::syncCallback(
  const sensor_msgs::msg::Image::ConstSharedPtr & rgb,
  const sensor_msgs::msg::Image::ConstSharedPtr & depth,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & cam_info,
  const sensor_msgs::msg::Image::ConstSharedPtr & mask)
{
  // Run processFrame() off-executor: it cv-waits for onRefineResult/onScoreResult,
  // which are dispatched by this same executor -> deadlock if run inline.
  if (processing_.exchange(true)) {
    return;
  }
  if (processing_thread_.joinable()) {
    processing_thread_.join();
  }
  processing_thread_ = std::thread(
    &FoundationPoseNode::processFrame, this, rgb, depth, cam_info, mask);
}

void FoundationPoseNode::processFrame(
  sensor_msgs::msg::Image::ConstSharedPtr rgb,
  sensor_msgs::msg::Image::ConstSharedPtr depth,
  sensor_msgs::msg::CameraInfo::ConstSharedPtr cam_info,
  sensor_msgs::msg::Image::ConstSharedPtr mask)
{
  std::lock_guard<std::mutex> mesh_lock(mesh_mutex_);
  auto mesh_data = mesh_loader_->getMeshData();
  if (!mesh_data || mesh_data->num_vertices == 0) {
    RCLCPP_WARN(get_logger(), "Mesh not loaded, skipping frame");
    processing_.store(false);
    return;
  }

  const ImageDim frame_size{rgb->height, rgb->width};
  const bool camera_size_matches =
    (cam_info->width == 0 || cam_info->width == rgb->width) &&
    (cam_info->height == 0 || cam_info->height == rgb->height);
  if (rgb->encoding != sensor_msgs::image_encodings::RGB8 ||
    depth->encoding != sensor_msgs::image_encodings::TYPE_32FC1 ||
    mask->encoding != sensor_msgs::image_encodings::MONO8 ||
    depth->height != rgb->height || depth->width != rgb->width ||
    mask->height != rgb->height || mask->width != rgb->width ||
    rgb->step < static_cast<size_t>(rgb->width) * 3 ||
    depth->step < static_cast<size_t>(depth->width) * sizeof(float) ||
    mask->step < mask->width || !camera_size_matches)
  {
    RCLCPP_ERROR(get_logger(), "Invalid FoundationPose RGB/depth/mask camera frame");
    processing_.store(false);
    return;
  }

  const uint32_t rgb_w = frame_size.width;
  const uint32_t H = resized_image_height_;
  const uint32_t W = resized_image_width_;
  constexpr int32_t C = 6;

  auto rgb_handle = cuda_buffer_backend::from_input_buffer(rgb->data, cuda_stream_);
  auto depth_handle = cuda_buffer_backend::from_input_buffer(depth->data, cuda_stream_);
  auto mask_handle = cuda_buffer_backend::from_input_buffer(mask->data, cuda_stream_);

  if (!pc_gpu_ || !(pc_size_ == frame_size)) {
    if (pc_gpu_) {
      cudaFree(pc_gpu_);
      pc_gpu_ = nullptr;
    }
    const size_t pc_floats = frame_size.numPixels() * 3;
    CHECK_CUDA_ERROR(cudaMalloc(&pc_gpu_, pc_floats * sizeof(float)), "malloc pc_gpu");
    pc_size_ = frame_size;
  }

  FrameObservationView frame{
    {rgb_handle.get_ptr(), frame_size, rgb->step, 3},
    {reinterpret_cast<const float *>(depth_handle.get_ptr()), frame_size, depth->step, 1},
    {mask_handle.get_ptr(), frame_size, mask->step, 1},
    {pc_gpu_, frame_size, static_cast<size_t>(rgb_w) * 3 * sizeof(float), 3},
    {static_cast<float>(cam_info->k[0]), static_cast<float>(cam_info->k[4]),
      static_cast<float>(cam_info->k[2]), static_cast<float>(cam_info->k[5])}};
  if (frame.rgb.row_stride_bytes < frame.rgb.minimumRowBytes() ||
    frame.depth.row_stride_bytes < frame.depth.minimumRowBytes() ||
    frame.mask.row_stride_bytes < frame.mask.minimumRowBytes() ||
    frame.intrinsics.fx <= 0.0f || frame.intrinsics.fy <= 0.0f)
  {
    RCLCPP_ERROR(get_logger(), "Invalid FoundationPose frame strides or intrinsics");
    processing_.store(false);
    return;
  }

  nvidia::isaac_ros::depth_to_xyz_map(
    cuda_stream_, frame.depth, frame.point_cloud, frame.intrinsics);
  CHECK_CUDA_ERROR(cudaGetLastError(), "depth_to_xyz_map");

  auto sampling = sampler_->sample(frame, mesh_data);
  const MeshGpuView mesh_view = MeshGpuView::from(*mesh_data);
  const RosMessageStamp stamp{
    rgb->header.frame_id, rgb->header.stamp.sec, rgb->header.stamp.nanosec};

  if (sampling.total_poses == 0) {
    RCLCPP_WARN(get_logger(), "Sampling produced 0 poses");
    processing_.store(false);
    return;
  }

  size_t all_poses_bytes = sampling.total_poses * 16 * sizeof(float);
  CHECK_CUDA_ERROR(cudaMemcpyAsync(all_poses_gpu_, sampling.poses.data(),
    all_poses_bytes, cudaMemcpyHostToDevice, cuda_stream_), "h2d poses");

  for (int iter = 0; iter < refine_iterations_; iter++) {
    for (int b = 0; b < sampling.num_batches; b++) {
      float * batch_ptr = all_poses_gpu_ + b * sampling.batch_size * 16;

      const std::vector<int64_t> refine_shape = {
        sampling.batch_size, H, W, C};
      auto t_rendered = TrtConv::allocate_tensor(refine_shape, kFloatCode, 32);
      auto t_observed = TrtConv::allocate_tensor(refine_shape, kFloatCode, 32);
      {
        auto wh1 = TrtConv::from_output_tensor(
          refine_input_tensor_names_[0], t_rendered, cuda_stream_);
        auto wh2 = TrtConv::from_output_tensor(
          refine_input_tensor_names_[1], t_observed, cuda_stream_);
        renderer_->renderRefine(
          {batch_ptr, static_cast<uint32_t>(sampling.batch_size)},
          frame, mesh_view,
          {reinterpret_cast<float *>(wh1.data()),
            reinterpret_cast<float *>(wh2.data()),
            static_cast<uint32_t>(sampling.batch_size), {H, W}});
      }

      TensorMsg::TensorListMsg trt_input;
      trt_input.header = rgb->header;
      trt_input.names = refine_input_tensor_names_;
      trt_input.tensors = {std::move(t_rendered), std::move(t_observed)};

      auto refine_result = callRefineTRT(std::move(trt_input));

      if (refine_result.tensors.size() < 2) {
        RCLCPP_ERROR(get_logger(), "Refine TRT returned %zu tensors, expected 2",
          refine_result.tensors.size());
        break;
      }

      const std::string trans_name = refine_result.names.size() > 0 ?
        refine_result.names[0] : "output_tensor1";
      const std::string rot_name = refine_result.names.size() > 1 ?
        refine_result.names[1] : "output_tensor2";
      auto trans_handle = TrtConv::from_input_tensor(
        trans_name, refine_result.tensors[0], cuda_stream_);
      auto rot_handle = TrtConv::from_input_tensor(
        rot_name, refine_result.tensors[1], cuda_stream_);

      transformer_->applyDeltas(
        {batch_ptr, static_cast<uint32_t>(sampling.batch_size)},
        {reinterpret_cast<const float *>(trans_handle.data()),
          reinterpret_cast<const float *>(rot_handle.data()),
          static_cast<uint32_t>(sampling.batch_size)},
        mesh_view);
    }
  }

  const std::vector<int64_t> score_shape = {
    sampling.total_poses, H, W, C};
  const size_t per_batch_floats =
    static_cast<size_t>(sampling.batch_size) * H * W * C;
  auto t_score_rendered = TrtConv::allocate_tensor(score_shape, kFloatCode, 32);
  auto t_score_observed = TrtConv::allocate_tensor(score_shape, kFloatCode, 32);
  {
    auto wh1 = TrtConv::from_output_tensor(
      score_input_tensor_names_[0], t_score_rendered, cuda_stream_);
    auto wh2 = TrtConv::from_output_tensor(
      score_input_tensor_names_[1], t_score_observed, cuda_stream_);
    float * sr_base = reinterpret_cast<float *>(wh1.data());
    float * so_base = reinterpret_cast<float *>(wh2.data());

    for (int b = 0; b < sampling.num_batches; b++) {
      float * batch_ptr = all_poses_gpu_ + b * sampling.batch_size * 16;
      score_renderer_->renderRefine(
        {batch_ptr, static_cast<uint32_t>(sampling.batch_size)},
        frame, mesh_view,
        {sr_base + b * per_batch_floats,
          so_base + b * per_batch_floats,
          static_cast<uint32_t>(sampling.batch_size), {H, W}});
    }
  }

  TensorMsg::TensorListMsg score_trt_input;
  score_trt_input.header = rgb->header;
  score_trt_input.names = score_input_tensor_names_;
  score_trt_input.tensors = {
    std::move(t_score_rendered), std::move(t_score_observed)};

  auto score_result = callScoreTRT(std::move(score_trt_input));

  if (score_result.tensors.empty()) {
    RCLCPP_ERROR(get_logger(), "Score TRT returned 0 tensors");
    processing_.store(false);
    return;
  }

  const std::string scores_name = score_result.names.empty() ?
    "output_tensor" : score_result.names[0];
  auto scores_handle = TrtConv::from_input_tensor(
    scores_name, score_result.tensors[0], cuda_stream_);

  auto result = decoder_->decode(
    {all_poses_gpu_, static_cast<uint32_t>(sampling.total_poses)},
    {reinterpret_cast<const float *>(scores_handle.data()),
      static_cast<uint32_t>(sampling.total_poses)},
    mesh_data, stamp);

  detection_pub_->publish(result.detection3d_array);

  // Broadcast TF
  if (!result.detection3d_array.detections.empty()) {
    const auto & det = result.detection3d_array.detections.front();
    geometry_msgs::msg::TransformStamped tf;
    tf.header = result.detection3d_array.header;
    tf.child_frame_id = tf_frame_name_;
    tf.transform.translation.x = det.bbox.center.position.x;
    tf.transform.translation.y = det.bbox.center.position.y;
    tf.transform.translation.z = det.bbox.center.position.z;
    tf.transform.rotation = det.bbox.center.orientation;
    tf_broadcaster_->sendTransform(tf);
  }

  {
    auto pose_tensor = TrtConv::allocate_tensor({1, 4, 4}, kFloatCode, 32);
    {
      auto wh = TrtConv::from_output_tensor("output", pose_tensor, cuda_stream_);
      CHECK_CUDA_ERROR(cudaMemcpyAsync(
          wh.data(), result.pose_matrix.data(), 16 * sizeof(float),
          cudaMemcpyHostToDevice, cuda_stream_), "h2d pose out");
    }

    TensorMsg::TensorListMsg pose_tl;
    pose_tl.header = rgb->header;
    pose_tl.names = {"output"};
    pose_tl.tensors = {std::move(pose_tensor)};
    pose_matrix_pub_->publish(std::move(pose_tl));
  }

  processing_.store(false);
}

}  // namespace foundationpose
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::foundationpose::FoundationPoseNode)
