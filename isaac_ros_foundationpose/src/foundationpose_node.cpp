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
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_shape.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_data_type.hpp"
#include "rcl_yaml_param_parser/parser.h"
#include "rclcpp_components/register_node_macro.hpp"
#include "std_msgs/msg/header.hpp"
#include "foundationpose_sampling.cu.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace foundationpose
{

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

  rgb_sub_ = std::make_shared<message_filters::Subscriber<nitros::NitrosImage>>(
    this, "pose_estimation/image", color_qos.get_rmw_qos_profile());
  depth_sub_ = std::make_shared<message_filters::Subscriber<nitros::NitrosImage>>(
    this, "pose_estimation/depth_image", depth_qos.get_rmw_qos_profile());
  cam_info_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::CameraInfo>>(
    this, "pose_estimation/camera_info", color_info_qos.get_rmw_qos_profile());
  mask_sub_ = std::make_shared<message_filters::Subscriber<nitros::NitrosImage>>(
    this, "pose_estimation/segmentation", segmentation_qos.get_rmw_qos_profile());

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
  pose_matrix_pub_ = create_publisher<nitros::NitrosTensorList>(
    "pose_estimation/pose_matrix_output", output_qos, pose_pub_options);

  // TRT topic publishers (FP node -> TRT nodes). Plain rclcpp publishers; the
  // NitrosTensorList type adapter still gives zero-copy GPU delivery to
  // TensorRTNode.
  rclcpp::PublisherOptions trt_pub_options;
  trt_pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  refine_pub_ = create_publisher<nitros::NitrosTensorList>(
    "refine/tensor_pub", rclcpp::QoS(1), trt_pub_options);
  score_pub_ = create_publisher<nitros::NitrosTensorList>(
    "score/tensor_pub", rclcpp::QoS(1), trt_pub_options);

  // Dedicated reentrant callback group for TRT result subscribers so their
  // dispatch is not blocked behind syncCallback / parameter callbacks on the
  // default callback group when the executor is busy.
  trt_callback_group_ = create_callback_group(
    rclcpp::CallbackGroupType::Reentrant);
  rclcpp::SubscriptionOptions trt_sub_options;
  trt_sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  trt_sub_options.callback_group = trt_callback_group_;

  refine_sub_ = create_subscription<nitros::NitrosTensorList>(
    "refine/tensor_sub", rclcpp::QoS(1),
    std::bind(&FoundationPoseNode::onRefineResult, this, std::placeholders::_1),
    trt_sub_options);
  score_sub_ = create_subscription<nitros::NitrosTensorList>(
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

  // Initialize NITROS-tensor pools. Each tensor has shape [N, H, W, 6] floats
  // (rendered RGB+XYZ or observed RGB+XYZ concatenated as 6 channels).
  const uint32_t H = resized_image_height_;
  const uint32_t W = resized_image_width_;
  constexpr uint32_t kChan = 6;
  const int32_t kNumBatches = 6;
  const uint32_t refine_batch = static_cast<uint32_t>(max_hypothesis_) / kNumBatches;
  const size_t refine_block_bytes = refine_batch * H * W * kChan * sizeof(float);
  const size_t score_block_bytes =
    static_cast<size_t>(max_hypothesis_) * H * W * kChan * sizeof(float);

  // 8 blocks: 2 per render step plus recycle headroom (QoS-1 TRT link).
  if (refine_pool_.create(refine_block_bytes, 8,
      nitros::CUDAMemoryPool::MemoryType::Device) != cudaSuccess)
  {
    throw std::runtime_error("[FoundationPoseNode] refine pool create failed");
  }
  // 4 blocks: in-flight frame plus recycle headroom.
  if (score_pool_.create(score_block_bytes, 4,
      nitros::CUDAMemoryPool::MemoryType::Device) != cudaSuccess)
  {
    throw std::runtime_error("[FoundationPoseNode] score pool create failed");
  }
  if (pose_matrix_pool_.create(16 * sizeof(float), 4,
      nitros::CUDAMemoryPool::MemoryType::Device) != cudaSuccess)
  {
    throw std::runtime_error("[FoundationPoseNode] pose_matrix pool create failed");
  }

  RCLCPP_INFO(get_logger(),
    "[FoundationPoseNode] Pipeline initialized "
    "(refine_pool=%zu MiB×8, score_pool=%zu MiB×4)",
    refine_block_bytes >> 20, score_block_bytes >> 20);
}

// --- Blocking TRT call helpers ---

void FoundationPoseNode::onRefineResult(
  const nitros::NitrosTensorList::ConstSharedPtr & result)
{
  std::lock_guard<std::mutex> lock(refine_mutex_);
  refine_result_ = *result;
  refine_result_ready_ = true;
  refine_cv_.notify_one();
}

void FoundationPoseNode::onScoreResult(
  const nitros::NitrosTensorList::ConstSharedPtr & result)
{
  std::lock_guard<std::mutex> lock(score_mutex_);
  score_result_ = *result;
  score_result_ready_ = true;
  score_cv_.notify_one();
}

nitros::NitrosTensorList FoundationPoseNode::callRefineTRT(
  nitros::NitrosTensorList input)
{
  constexpr auto kTimeout = std::chrono::seconds(5);
  std::unique_lock<std::mutex> lock(refine_mutex_);
  refine_result_ready_ = false;
  refine_pub_->publish(std::move(input));
  if (!refine_cv_.wait_for(lock, kTimeout, [this] {return refine_result_ready_;})) {
    RCLCPP_ERROR(get_logger(),
      "Refine TRT timed out after %" PRId64 "s; check that the TensorRT node is running",
      static_cast<int64_t>(kTimeout.count()));
    return nitros::NitrosTensorList(cuda_stream_);
  }
  return std::move(refine_result_);
}

nitros::NitrosTensorList FoundationPoseNode::callScoreTRT(
  nitros::NitrosTensorList input)
{
  constexpr auto kTimeout = std::chrono::seconds(5);
  std::unique_lock<std::mutex> lock(score_mutex_);
  score_result_ready_ = false;
  score_pub_->publish(std::move(input));
  if (!score_cv_.wait_for(lock, kTimeout, [this] {return score_result_ready_;})) {
    RCLCPP_ERROR(get_logger(),
      "Score TRT timed out after %" PRId64 "s; check that the TensorRT node is running",
      static_cast<int64_t>(kTimeout.count()));
    return nitros::NitrosTensorList(cuda_stream_);
  }
  return std::move(score_result_);
}

// --- Callbacks ---

void FoundationPoseNode::syncCallback(
  const nitros::NitrosImage::ConstSharedPtr & rgb,
  const nitros::NitrosImage::ConstSharedPtr & depth,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & cam_info,
  const nitros::NitrosImage::ConstSharedPtr & mask)
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
  nitros::NitrosImage::ConstSharedPtr rgb,
  nitros::NitrosImage::ConstSharedPtr depth,
  sensor_msgs::msg::CameraInfo::ConstSharedPtr cam_info,
  nitros::NitrosImage::ConstSharedPtr mask)
{
  std::lock_guard<std::mutex> mesh_lock(mesh_mutex_);
  auto mesh_data = mesh_loader_->getMeshData();
  if (!mesh_data || mesh_data->num_vertices == 0) {
    RCLCPP_WARN(get_logger(), "Mesh not loaded, skipping frame");
    processing_.store(false);
    return;
  }

  const uint32_t rgb_h = rgb->height;
  const uint32_t rgb_w = rgb->width;
  const uint32_t H = resized_image_height_;
  const uint32_t W = resized_image_width_;
  constexpr int32_t C = 6;

  Eigen::Matrix3f K;
  K << static_cast<float>(cam_info->k[0]), 0.0f, static_cast<float>(cam_info->k[2]),
    0.0f, static_cast<float>(cam_info->k[4]), static_cast<float>(cam_info->k[5]),
    0.0f, 0.0f, 1.0f;

  auto rgb_handle = rgb->get_read_handle(cuda_stream_);
  auto depth_handle = depth->get_read_handle(cuda_stream_);
  auto mask_handle = mask->get_read_handle(cuda_stream_);

  if (!pc_gpu_) {
    const size_t pc_floats = static_cast<size_t>(rgb_h) * rgb_w * 3;
    CHECK_CUDA_ERROR(cudaMalloc(&pc_gpu_, pc_floats * sizeof(float)), "malloc pc_gpu");
  }
  nvidia::isaac_ros::depth_to_xyz_map(
    cuda_stream_, reinterpret_cast<const float *>(depth_handle.get_ptr()),
    pc_gpu_, rgb_h, rgb_w,
    static_cast<float>(cam_info->k[0]), static_cast<float>(cam_info->k[4]),
    static_cast<float>(cam_info->k[2]), static_cast<float>(cam_info->k[5]));
  CHECK_CUDA_ERROR(cudaGetLastError(), "depth_to_xyz_map");

  auto sampling = sampler_->sample(
    reinterpret_cast<const float *>(depth_handle.get_ptr()),
    mask_handle.get_ptr(), rgb_h, rgb_w, K, mesh_data);

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

      // Acquire two pool blocks and have the renderer write directly into
      // them. WriteHandles are kept alive across renderRefine so their dtor
      // records the completion event AFTER the renderer's queued kernels.
      const nitros::NitrosTensorShape refine_shape({sampling.batch_size, H, W, C});
      nitros::NitrosTensor t_rendered, t_observed;
      {
        auto wh1 = t_rendered.from_pool(
          refine_input_tensor_names_[0], refine_pool_,
          refine_shape, nitros::NitrosDataType::kFloat32, cuda_stream_);
        auto wh2 = t_observed.from_pool(
          refine_input_tensor_names_[1], refine_pool_,
          refine_shape, nitros::NitrosDataType::kFloat32, cuda_stream_);
        renderer_->renderRefine(
          batch_ptr, sampling.batch_size,
          pc_gpu_, rgb_handle.get_ptr(), K, rgb_h, rgb_w, mesh_data,
          reinterpret_cast<float *>(wh1.get_ptr()),
          reinterpret_cast<float *>(wh2.get_ptr()));
      }  // dtors record completion event on cuda_stream_

      nitros::NitrosTensorList trt_input(cuda_stream_);
      trt_input.set_timestamp_sec(rgb->timestamp_sec);
      trt_input.set_timestamp_nsec(rgb->timestamp_nsec);
      trt_input.set_frame_id(rgb->frame_id);
      trt_input.add_tensor(std::move(t_rendered));
      trt_input.add_tensor(std::move(t_observed));

      auto refine_result = callRefineTRT(std::move(trt_input));

      if (refine_result.num_tensors() < 2) {
        RCLCPP_ERROR(get_logger(), "Refine TRT returned %zu tensors, expected 2",
          refine_result.num_tensors());
        break;
      }

      auto trans_handle = refine_result.get_read_handle(cuda_stream_, 0);
      auto rot_handle = refine_result.get_read_handle(cuda_stream_, 1);

      transformer_->applyDeltas(
        batch_ptr, sampling.batch_size,
        const_cast<void *>(static_cast<const void *>(trans_handle.get_ptr())),
        const_cast<void *>(static_cast<const void *>(rot_handle.get_ptr())),
        mesh_data);
    }
  }

  // Acquire two score pool blocks; have the renderer write each batch's
  // output into consecutive offsets of the same buffer pair. Hold the
  // WriteHandles across all batches so the completion event covers all
  // queued kernels.
  const nitros::NitrosTensorShape score_shape(
    {sampling.total_poses, H, W, C});
  const size_t per_batch_floats =
    static_cast<size_t>(sampling.batch_size) * H * W * C;
  nitros::NitrosTensor t_score_rendered, t_score_observed;
  {
    auto wh1 = t_score_rendered.from_pool(
      score_input_tensor_names_[0], score_pool_,
      score_shape, nitros::NitrosDataType::kFloat32, cuda_stream_);
    auto wh2 = t_score_observed.from_pool(
      score_input_tensor_names_[1], score_pool_,
      score_shape, nitros::NitrosDataType::kFloat32, cuda_stream_);
    float * sr_base = reinterpret_cast<float *>(wh1.get_ptr());
    float * so_base = reinterpret_cast<float *>(wh2.get_ptr());

    for (int b = 0; b < sampling.num_batches; b++) {
      float * batch_ptr = all_poses_gpu_ + b * sampling.batch_size * 16;
      score_renderer_->renderRefine(
        batch_ptr, sampling.batch_size,
        pc_gpu_, rgb_handle.get_ptr(), K, rgb_h, rgb_w, mesh_data,
        sr_base + b * per_batch_floats,
        so_base + b * per_batch_floats);
    }
  }  // dtors record completion event on cuda_stream_

  nitros::NitrosTensorList score_trt_input(cuda_stream_);
  score_trt_input.set_timestamp_sec(rgb->timestamp_sec);
  score_trt_input.set_timestamp_nsec(rgb->timestamp_nsec);
  score_trt_input.set_frame_id(rgb->frame_id);
  score_trt_input.add_tensor(std::move(t_score_rendered));
  score_trt_input.add_tensor(std::move(t_score_observed));

  auto score_result = callScoreTRT(std::move(score_trt_input));

  if (score_result.num_tensors() == 0) {
    RCLCPP_ERROR(get_logger(), "Score TRT returned 0 tensors");
    processing_.store(false);
    return;
  }

  auto scores_handle = score_result.get_read_handle(cuda_stream_, 0);

  auto result = decoder_->decode(
    all_poses_gpu_, sampling.total_poses,
    reinterpret_cast<const float *>(scores_handle.get_ptr()),
    mesh_data, rgb->frame_id, rgb->timestamp_sec, rgb->timestamp_nsec);

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

  // Publish pose matrix (4x4) as NITROS via from_pool. The pool ref-counts
  // the buffer so subscribers (Selector / NitrosPlaybackNode) can hold the
  // message arbitrarily long without racing the next frame's write.
  {
    nitros::NitrosTensor pose_tensor;
    {
      auto wh = pose_tensor.from_pool(
        "output", pose_matrix_pool_,
        nitros::NitrosTensorShape({1, 4, 4}),
        nitros::NitrosDataType::kFloat32,
        cuda_stream_);
      CHECK_CUDA_ERROR(cudaMemcpyAsync(
          wh.get_ptr(), result.pose_matrix.data(), 16 * sizeof(float),
          cudaMemcpyHostToDevice, cuda_stream_), "h2d pose out");
    }  // wh dtor records event after the memcpy

    nitros::NitrosTensorList pose_tl(cuda_stream_);
    pose_tl.set_timestamp_sec(rgb->timestamp_sec);
    pose_tl.set_timestamp_nsec(rgb->timestamp_nsec);
    pose_tl.set_frame_id(rgb->frame_id);
    pose_tl.add_tensor(std::move(pose_tensor));
    pose_matrix_pub_->publish(std::move(pose_tl));
  }

  processing_.store(false);
}

}  // namespace foundationpose
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::foundationpose::FoundationPoseNode)
