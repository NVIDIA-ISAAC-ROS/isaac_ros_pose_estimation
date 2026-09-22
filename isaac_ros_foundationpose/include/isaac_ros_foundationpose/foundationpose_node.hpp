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

#ifndef ISAAC_ROS_FOUNDATIONPOSE__FOUNDATIONPOSE_NODE_HPP_
#define ISAAC_ROS_FOUNDATIONPOSE__FOUNDATIONPOSE_NODE_HPP_

#include <cuda_runtime.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "isaac_ros_tensor_msgs/tensor_list_msg.hpp"
#include "tensorrt_conversions/tensorrt_conversions.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"

#include "message_filters/subscriber.hpp"
#include "message_filters/synchronizer.hpp"
#include "message_filters/sync_policies/approximate_time.hpp"

#include "isaac_ros_foundationpose/foundationpose_impl/mesh_loader.hpp"
#include "isaac_ros_foundationpose/foundationpose_impl/pose_sampler.hpp"
#include "isaac_ros_foundationpose/foundationpose_impl/pose_renderer.hpp"
#include "isaac_ros_foundationpose/foundationpose_impl/pose_transformer.hpp"
#include "isaac_ros_foundationpose/foundationpose_impl/pose_decoder.hpp"
#include "isaac_ros_foundationpose/srv/switch_mesh.hpp"

using StringList = std::vector<std::string>;

namespace nvidia
{
namespace isaac_ros
{
namespace foundationpose
{

namespace TensorMsg = nvidia::isaac_ros::isaac_ros_tensor_msgs;
namespace TrtConv = nvidia::isaac_ros::tensorrt_conversions;

class FoundationPoseNode : public rclcpp::Node
{
public:
  explicit FoundationPoseNode(rclcpp::NodeOptions options = rclcpp::NodeOptions());
  ~FoundationPoseNode() override;

private:
  void syncCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr & rgb,
    const sensor_msgs::msg::Image::ConstSharedPtr & depth,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & cam_info,
    const sensor_msgs::msg::Image::ConstSharedPtr & mask);

  void processFrame(
    sensor_msgs::msg::Image::ConstSharedPtr rgb,
    sensor_msgs::msg::Image::ConstSharedPtr depth,
    sensor_msgs::msg::CameraInfo::ConstSharedPtr cam_info,
    sensor_msgs::msg::Image::ConstSharedPtr mask);

  void initializePipeline();

  // Blocking TRT call helpers
  TensorMsg::TensorListMsg callRefineTRT(TensorMsg::TensorListMsg input);
  TensorMsg::TensorListMsg callScoreTRT(TensorMsg::TensorListMsg input);

  void onRefineResult(const TensorMsg::TensorListMsg::ConstSharedPtr & result);
  void onScoreResult(const TensorMsg::TensorListMsg::ConstSharedPtr & result);

  // Input sync
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image, sensor_msgs::msg::Image,
    sensor_msgs::msg::CameraInfo, sensor_msgs::msg::Image>;

  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> rgb_sub_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> depth_sub_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::CameraInfo>> cam_info_sub_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> mask_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  // Output publishers
  rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr detection_pub_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr reset_client_;
  rclcpp::Client<isaac_ros_foundationpose::srv::SwitchMesh>::SharedPtr
    tracking_switch_mesh_client_;
  rclcpp::Publisher<TensorMsg::TensorListMsg>::SharedPtr pose_matrix_pub_;

  // TRT topic pub/sub (blocking pattern)
  rclcpp::Publisher<TensorMsg::TensorListMsg>::SharedPtr refine_pub_;
  rclcpp::Subscription<TensorMsg::TensorListMsg>::SharedPtr refine_sub_;
  rclcpp::Publisher<TensorMsg::TensorListMsg>::SharedPtr score_pub_;
  rclcpp::Subscription<TensorMsg::TensorListMsg>::SharedPtr score_sub_;
  rclcpp::CallbackGroup::SharedPtr trt_callback_group_;

  // Blocking sync state for refine TRT
  std::mutex refine_mutex_;
  std::condition_variable refine_cv_;
  bool refine_result_ready_{false};
  TensorMsg::TensorListMsg refine_result_;

  // Blocking sync state for score TRT
  std::mutex score_mutex_;
  std::condition_variable score_cv_;
  bool score_result_ready_{false};
  TensorMsg::TensorListMsg score_result_;

  // Single-frame processing gate and watchdog
  std::atomic<bool> processing_{false};
  std::thread processing_thread_;
  rclcpp::TimerBase::SharedPtr watchdog_timer_;
  int64_t pose_estimation_timeout_ms_{5000};

  // CUDA stream
  cudaStream_t cuda_stream_{nullptr};

  // Pre-allocated GPU buffers (no cudaMalloc/cudaFree in callbacks)
  float * all_poses_gpu_{nullptr};         // [max_hypothesis, 4, 4] sampled/refined poses
  float * pc_gpu_{nullptr};                // [rgb_h * rgb_w * 3] lazy-alloc on first frame
  ImageDim pc_size_;

  // Pipeline components (no TensorRT -- that is in separate nodes)
  std::unique_ptr<MeshLoader> mesh_loader_;
  std::unique_ptr<PoseSampler> sampler_;
  std::unique_ptr<PoseRenderer> renderer_;
  std::unique_ptr<PoseRenderer> score_renderer_;
  std::unique_ptr<PoseTransformer> transformer_;
  std::unique_ptr<PoseDecoder> decoder_;

  // Parameters
  std::string configuration_file_;
  std::string mesh_file_path_;
  StringList refine_input_tensor_names_;
  StringList score_input_tensor_names_;
  float min_depth_{0.1f};
  float max_depth_{2.0f};
  int32_t refine_iterations_{3};
  float rot_normalizer_{1.0f};
  uint32_t resized_image_width_{160};
  uint32_t resized_image_height_{160};
  float refine_crop_ratio_{1.2f};
  float score_crop_ratio_{1.2f};
  int32_t max_hypothesis_{252};

  // TF broadcast
  std::string tf_frame_name_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::mutex param_mutex_;
  std::mutex mesh_mutex_;

  // Dynamic parameter callbacks
  std::shared_ptr<rclcpp::ParameterEventHandler> param_subscriber_;
  std::shared_ptr<rclcpp::ParameterCallbackHandle> mesh_file_path_cb_;
  std::shared_ptr<rclcpp::ParameterCallbackHandle> tf_frame_name_cb_;
  std::shared_ptr<rclcpp::ParameterCallbackHandle> fixed_translations_cb_;
  std::shared_ptr<rclcpp::ParameterCallbackHandle> fixed_axis_angles_cb_;
  std::shared_ptr<rclcpp::ParameterCallbackHandle> symmetry_axes_cb_;
  std::shared_ptr<rclcpp::ParameterCallbackHandle> symmetry_planes_cb_;

  // Sampling constraint params
  StringList symmetry_axes_;
  StringList symmetry_planes_;
  StringList fixed_axis_angles_;
  StringList fixed_translations_;
};

}  // namespace foundationpose
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_FOUNDATIONPOSE__FOUNDATIONPOSE_NODE_HPP_
