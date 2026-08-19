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

#include "isaac_ros_foundationpose/foundationpose_tracking_node.hpp"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "foundationpose_render.cu.hpp"
#include "foundationpose_sampling.cu.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_shape.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_data_type.hpp"
#include "rcl_yaml_param_parser/parser.h"
#include "rclcpp_components/register_node_macro.hpp"
#include "std_msgs/msg/header.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace foundationpose
{

FoundationPoseTrackingNode::FoundationPoseTrackingNode(rclcpp::NodeOptions options)
: rclcpp::Node("foundationpose_tracking_node", options),
  configuration_file_(
    declare_parameter<std::string>("configuration_file", "foundationpose_model_config.yaml")),
  mesh_file_path_(declare_parameter<std::string>("mesh_file_path", "")),
  refine_input_tensor_names_(declare_parameter<StringList>(
      "refine_input_tensor_names", StringList{"input_tensor1", "input_tensor2"})),
  min_depth_(declare_parameter<double>("min_depth", 0.1)),
  max_depth_(declare_parameter<double>("max_depth", 2.0)),
  tf_frame_name_(declare_parameter<std::string>("tf_frame_name", "fp_object")),
  min_pointcloud_support_(declare_parameter<int>("min_pointcloud_support", 50)),
  enable_auto_reset_(declare_parameter<bool>("enable_auto_reset", true))
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

  rcl_variant_t * rot_normalizer = rcl_yaml_node_struct_get(
    "foundationpose", "rot_normalizer", foundationpose_params);
  if (!rot_normalizer->double_value) {
    RCLCPP_ERROR(get_logger(), "No rot_normalizer parameter found");
    throw std::runtime_error("Parameter parsing failure.");
  }
  rot_normalizer_ = static_cast<float>(*rot_normalizer->double_value);

  rcl_yaml_node_struct_fini(foundationpose_params);

  initializePipeline();

  rclcpp::QoS output_qos = ::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos");
  // Per-channel input QoS, matching the GXF FoundationposeTrackingNode parameter
  // names so the manipulator's launch keeps working.
  rclcpp::QoS color_qos =
    ::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "color_qos", 10);
  rclcpp::QoS depth_qos =
    ::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "depth_qos", 10);
  rclcpp::QoS color_info_qos =
    ::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "color_info_qos", 10);

  rgb_sub_ = std::make_shared<message_filters::Subscriber<nitros::NitrosImage>>(
    this, "tracking/image", color_qos.get_rmw_qos_profile());
  depth_sub_ = std::make_shared<message_filters::Subscriber<nitros::NitrosImage>>(
    this, "tracking/depth_image", depth_qos.get_rmw_qos_profile());
  cam_info_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::CameraInfo>>(
    this, "tracking/camera_info", color_info_qos.get_rmw_qos_profile());

  const size_t sync_depth = std::max(
    {color_qos.depth(), depth_qos.depth(), color_info_qos.depth()});
  sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
    SyncPolicy(sync_depth), *rgb_sub_, *depth_sub_, *cam_info_sub_);
  sync_->registerCallback(
    std::bind(
      &FoundationPoseTrackingNode::syncCallback, this,
      std::placeholders::_1, std::placeholders::_2,
      std::placeholders::_3));

  pose_state_sub_ = create_subscription<isaac_ros_tensor_list_interfaces::msg::TensorList>(
    "tracking/pose_input", rclcpp::QoS(10),
    [this](const isaac_ros_tensor_list_interfaces::msg::TensorList::ConstSharedPtr & msg) {
      std::lock_guard<std::mutex> lock(pose_state_mutex_);
      latest_pose_input_ = msg;
    });

  detection_pub_ = create_publisher<vision_msgs::msg::Detection3DArray>(
    "tracking/output", output_qos);
  reset_client_ = create_client<std_srvs::srv::Trigger>("selector/reset");
  switch_mesh_srv_ = create_service<isaac_ros_foundationpose::srv::SwitchMesh>(
    "tracking/switch_mesh",
    [this](
      const std::shared_ptr<isaac_ros_foundationpose::srv::SwitchMesh::Request> request,
      std::shared_ptr<isaac_ros_foundationpose::srv::SwitchMesh::Response> response) {
      response->success = switchMesh(request->mesh_file_path, true, response->message);
    });

  rclcpp::PublisherOptions pose_pub_options;
  pose_pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  pose_matrix_pub_ = create_publisher<nitros::NitrosTensorList>(
    "tracking/pose_matrix_output", output_qos, pose_pub_options);

  rclcpp::PublisherOptions refine_pub_options;
  refine_pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  refine_pub_ = create_publisher<nitros::NitrosTensorList>(
    "tracking_refine/tensor_pub", rclcpp::QoS(1), refine_pub_options);

  // Dedicated reentrant callback group for the TRT result subscriber so its
  // dispatch is not blocked behind syncCallback / parameter callbacks /
  // pose_state_sub on the default callback group when the executor is busy.
  trt_callback_group_ = create_callback_group(
    rclcpp::CallbackGroupType::Reentrant);
  rclcpp::SubscriptionOptions refine_sub_options;
  refine_sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  refine_sub_options.callback_group = trt_callback_group_;
  refine_sub_ = create_subscription<nitros::NitrosTensorList>(
    "tracking_refine/tensor_sub", rclcpp::QoS(1),
    std::bind(&FoundationPoseTrackingNode::onRefineResult, this, std::placeholders::_1),
    refine_sub_options);

  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  worker_thread_ = std::thread(&FoundationPoseTrackingNode::workerLoop, this);

  RCLCPP_INFO(get_logger(), "[FoundationPoseTrackingNode] Initialization complete");
}

bool FoundationPoseTrackingNode::switchMesh(
  const std::string & mesh_file_path, bool request_selector_reset, std::string & message)
{
  try {
    {
      std::lock_guard<std::mutex> lock(mesh_mutex_);
      mesh_file_path_ = mesh_file_path;
      mesh_loader_->tryReload(mesh_file_path_);
    }
    {
      std::lock_guard<std::mutex> pose_lock(pose_state_mutex_);
      latest_pose_input_.reset();
    }
    if (request_selector_reset) {
      if (reset_client_->service_is_ready()) {
        reset_client_->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>());
      } else {
        RCLCPP_WARN(
          get_logger(),
          "[FoundationPoseTrackingNode] selector/reset service is not ready; "
          "waiting for selector timeout");
      }
    }
    message = "mesh reloaded and tracking pose input cleared";
    RCLCPP_INFO(get_logger(), "[FoundationPoseTrackingNode] %s: %s",
      message.c_str(), mesh_file_path.c_str());
    return true;
  } catch (const std::exception & e) {
    message = std::string("failed to switch mesh: ") + e.what();
    return false;
  }
}

FoundationPoseTrackingNode::~FoundationPoseTrackingNode()
{
  {
    std::lock_guard<std::mutex> lock(work_mutex_);
    shutdown_ = true;
  }
  work_cv_.notify_one();
  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
  renderer_.reset();
  transformer_.reset();
  decoder_.reset();
  mesh_loader_.reset();
  if (pc_gpu_) {cudaFree(pc_gpu_);}
  if (pose_input_gpu_) {cudaFree(pose_input_gpu_);}
  // Pools self-destroy.
  if (support_count_device_) {cudaFree(support_count_device_);}
  if (cuda_stream_) {
    cudaStreamDestroy(cuda_stream_);
  }
}

void FoundationPoseTrackingNode::initializePipeline()
{
  cudaStreamCreate(&cuda_stream_);

  mesh_loader_ = std::make_unique<MeshLoader>(cuda_stream_);
  if (!mesh_file_path_.empty()) {
    mesh_loader_->load(mesh_file_path_);
  }

  PoseRendererParams render_params;
  render_params.crop_ratio = refine_crop_ratio_;
  render_params.min_depth = min_depth_;
  render_params.max_depth = max_depth_;
  render_params.resized_height = resized_image_height_;
  render_params.resized_width = resized_image_width_;
  renderer_ = std::make_unique<PoseRenderer>(render_params, cuda_stream_);

  transformer_ = std::make_unique<PoseTransformer>(rot_normalizer_, cuda_stream_);
  decoder_ = std::make_unique<PoseDecoder>(cuda_stream_);

  CHECK_CUDA_ERROR(cudaMalloc(&pose_input_gpu_, 16 * sizeof(float)), "malloc pose_input_gpu");

  const uint32_t H = resized_image_height_;
  const uint32_t W = resized_image_width_;
  constexpr uint32_t kChan = 6;
  const size_t refine_block_bytes = H * W * kChan * sizeof(float);

  // 8 blocks: 2 per frame plus recycle headroom (QoS-1 TRT link).
  if (refine_pool_.create(refine_block_bytes, 8,
      nitros::CUDAMemoryPool::MemoryType::Device) != cudaSuccess)
  {
    throw std::runtime_error("[FoundationPoseTrackingNode] refine pool create failed");
  }
  if (pose_matrix_pool_.create(16 * sizeof(float), 4,
      nitros::CUDAMemoryPool::MemoryType::Device) != cudaSuccess)
  {
    throw std::runtime_error("[FoundationPoseTrackingNode] pose_matrix pool create failed");
  }
  CHECK_CUDA_ERROR(cudaMalloc(&support_count_device_, sizeof(int)), "malloc support_count");

  if (auto md = mesh_loader_->getMeshData()) {
    const float max_pose_jump_m = md->mesh_diameter * refine_crop_ratio_ / 2.0f;
    RCLCPP_INFO(get_logger(),
      "[FoundationPoseTrackingNode] mesh_diameter=%.4f m, refine_crop_ratio=%.3f, "
      "max tolerable pose jump = %.4f m (recommended selector max_pose_jump_m)",
      md->mesh_diameter, refine_crop_ratio_, max_pose_jump_m);
  }

  RCLCPP_INFO(get_logger(), "[FoundationPoseTrackingNode] Pipeline initialized");
}

// --- Blocking TRT call helpers ---

void FoundationPoseTrackingNode::onRefineResult(
  const nitros::NitrosTensorList::ConstSharedPtr & result)
{
  std::lock_guard<std::mutex> lock(refine_mutex_);
  refine_result_ = *result;
  refine_result_ready_ = true;
  refine_cv_.notify_one();
}

nitros::NitrosTensorList FoundationPoseTrackingNode::callRefineTRT(
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

// --- Callbacks ---

void FoundationPoseTrackingNode::workerLoop()
{
  while (true) {
    std::function<void()> work;
    {
      std::unique_lock<std::mutex> lock(work_mutex_);
      work_cv_.wait(lock, [this] {return work_ready_ || shutdown_;});
      if (shutdown_) {return;}
      work = std::move(pending_work_);
      work_ready_ = false;
    }
    work();
  }
}

void FoundationPoseTrackingNode::syncCallback(
  const nitros::NitrosImage::ConstSharedPtr & rgb,
  const nitros::NitrosImage::ConstSharedPtr & depth,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & cam_info)
{
  isaac_ros_tensor_list_interfaces::msg::TensorList::ConstSharedPtr pose_input;
  {
    std::lock_guard<std::mutex> pose_lock(pose_state_mutex_);
    pose_input = latest_pose_input_;
  }
  if (!pose_input) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "[FP-track] no pose state yet, skipping frame");
    return;
  }
  std::lock_guard<std::mutex> lock(work_mutex_);
  if (work_ready_) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "[FP-track] dropping frame, worker busy");
    return;
  }
  pending_work_ = [this, rgb, depth, cam_info, pose_input]() {
      processFrame(rgb, depth, cam_info, pose_input);
    };
  work_ready_ = true;
  work_cv_.notify_one();
}

void FoundationPoseTrackingNode::processFrame(
  nitros::NitrosImage::ConstSharedPtr rgb,
  nitros::NitrosImage::ConstSharedPtr depth,
  sensor_msgs::msg::CameraInfo::ConstSharedPtr cam_info,
  isaac_ros_tensor_list_interfaces::msg::TensorList::ConstSharedPtr pose_input)
{
  std::lock_guard<std::mutex> mesh_lock(mesh_mutex_);
  auto mesh_data = mesh_loader_->getMeshData();
  if (!mesh_data || mesh_data->num_vertices == 0) {
    RCLCPP_WARN(get_logger(), "[FoundationPoseTrackingNode] Mesh not loaded, skipping");
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

  if (pose_input->tensors.empty() ||
    pose_input->tensors[0].data.size() < 16 * sizeof(float))
  {
    RCLCPP_WARN(get_logger(),
      "[FoundationPoseTrackingNode] Invalid pose tensor in input");
    return;
  }
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
      pose_input_gpu_, pose_input->tensors[0].data.data(),
      16 * sizeof(float), cudaMemcpyHostToDevice, cuda_stream_), "copy pose");
  // No sync needed: renderer/transformer below are launched on cuda_stream_.

  // Acquire pool blocks; renderer writes directly into them. WriteHandles
  // are alive across renderRefine so their dtor records the completion event
  // AFTER the renderer's queued kernels.
  const nitros::NitrosTensorShape refine_shape({1, H, W, C});
  nitros::NitrosTensor t_rendered, t_observed;
  {
    auto wh1 = t_rendered.from_pool(
      refine_input_tensor_names_[0], refine_pool_,
      refine_shape, nitros::NitrosDataType::kFloat32, cuda_stream_);
    auto wh2 = t_observed.from_pool(
      refine_input_tensor_names_[1], refine_pool_,
      refine_shape, nitros::NitrosDataType::kFloat32, cuda_stream_);
    renderer_->renderRefine(
      pose_input_gpu_, 1,
      pc_gpu_, rgb_handle.get_ptr(), K, rgb_h, rgb_w, mesh_data,
      reinterpret_cast<float *>(wh1.get_ptr()),
      reinterpret_cast<float *>(wh2.get_ptr()));
  }

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
    return;
  }

  auto trans_handle = refine_result.get_read_handle(cuda_stream_, 0);
  auto rot_handle = refine_result.get_read_handle(cuda_stream_, 1);

  transformer_->applyDeltas(
    pose_input_gpu_, 1,
    const_cast<void *>(static_cast<const void *>(trans_handle.get_ptr())),
    const_cast<void *>(static_cast<const void *>(rot_handle.get_ptr())),
    mesh_data);

  // Decode (tracking mode -- no scoring)
  auto result = decoder_->decodeTracking(
    pose_input_gpu_, mesh_data,
    rgb->frame_id, rgb->timestamp_sec, rgb->timestamp_nsec);

  // Quality gate: if the claimed pose has no point-cloud support, the tracker
  // is drifting on a ghost. Skip publishing so selector's watchdog resets.
  if (result.detection3d_array.detections.empty()) {
    return;
  }
  if (enable_auto_reset_) {
    const auto & pos = result.detection3d_array.detections.front().bbox.center.position;
    const float tx = static_cast<float>(pos.x);
    const float ty = static_cast<float>(pos.y);
    const float tz = static_cast<float>(pos.z);
    const float radius = mesh_data->mesh_diameter * 0.5f;
    const int n_points = static_cast<int>(rgb_h * rgb_w);
    CHECK_CUDA_ERROR(cudaMemsetAsync(support_count_device_, 0, sizeof(int), cuda_stream_),
      "zero support count");
    nvidia::isaac_ros::count_points_within_radius(
      cuda_stream_, pc_gpu_, n_points, tx, ty, tz, radius * radius, support_count_device_);
    int support = 0;
    CHECK_CUDA_ERROR(cudaMemcpyAsync(&support, support_count_device_, sizeof(int),
      cudaMemcpyDeviceToHost, cuda_stream_), "read support count");
    CHECK_CUDA_ERROR(cudaStreamSynchronize(cuda_stream_), "sync support count");
    if (support < min_pointcloud_support_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "[FP-track] pose has only %d pointcloud points within %.3f m, skipping publish",
        support, radius);
      return;
    }
  }

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

  // Publish pose matrix (4x4) for Selector feedback loop, via from_pool.
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
    }

    nitros::NitrosTensorList pose_tl(cuda_stream_);
    pose_tl.set_timestamp_sec(rgb->timestamp_sec);
    pose_tl.set_timestamp_nsec(rgb->timestamp_nsec);
    pose_tl.set_frame_id(rgb->frame_id);
    pose_tl.add_tensor(std::move(pose_tensor));
    pose_matrix_pub_->publish(std::move(pose_tl));
  }
}

}  // namespace foundationpose
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::foundationpose::FoundationPoseTrackingNode)
