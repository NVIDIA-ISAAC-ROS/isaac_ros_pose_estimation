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

#ifndef ISAAC_ROS_FOUNDATIONPOSE__FOUNDATIONPOSE_TRACKING_NODE_HPP_
#define ISAAC_ROS_FOUNDATIONPOSE__FOUNDATIONPOSE_TRACKING_NODE_HPP_

#include <cuda_runtime.h>

#include <condition_variable>
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
#include "isaac_ros_foundationpose/srv/switch_mesh.hpp"

#include "message_filters/subscriber.hpp"
#include "message_filters/synchronizer.hpp"
#include "message_filters/sync_policies/exact_time.hpp"

#include "isaac_ros_foundationpose/foundationpose_impl/mesh_loader.hpp"
#include "isaac_ros_foundationpose/foundationpose_impl/pose_renderer.hpp"
#include "isaac_ros_foundationpose/foundationpose_impl/pose_transformer.hpp"
#include "isaac_ros_foundationpose/foundationpose_impl/pose_decoder.hpp"

using StringList = std::vector<std::string>;

namespace nvidia
{
namespace isaac_ros
{
namespace foundationpose
{

namespace TensorMsg = nvidia::isaac_ros::isaac_ros_tensor_msgs;
namespace TrtConv = nvidia::isaac_ros::tensorrt_conversions;

class FoundationPoseTrackingNode : public rclcpp::Node
{
public:
  explicit FoundationPoseTrackingNode(rclcpp::NodeOptions options = rclcpp::NodeOptions());
  ~FoundationPoseTrackingNode() override;

private:
  void syncCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr & rgb,
    const sensor_msgs::msg::Image::ConstSharedPtr & depth,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & cam_info);

  void processFrame(
    sensor_msgs::msg::Image::ConstSharedPtr rgb,
    sensor_msgs::msg::Image::ConstSharedPtr depth,
    sensor_msgs::msg::CameraInfo::ConstSharedPtr cam_info,
    TensorMsg::TensorListMsg::ConstSharedPtr pose_input);

  void initializePipeline();

  TensorMsg::TensorListMsg callRefineTRT(TensorMsg::TensorListMsg input);
  void onRefineResult(const TensorMsg::TensorListMsg::ConstSharedPtr & result);
  bool switchMesh(
    const std::string & mesh_file_path, bool request_selector_reset,
    std::string & message);

  using SyncPolicy = message_filters::sync_policies::ExactTime<
    sensor_msgs::msg::Image, sensor_msgs::msg::Image,
    sensor_msgs::msg::CameraInfo>;

  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> rgb_sub_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> depth_sub_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::CameraInfo>> cam_info_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  // Pose input is state, not a synced data stream.
  rclcpp::Subscription<TensorMsg::TensorListMsg>::SharedPtr pose_state_sub_;
  std::mutex pose_state_mutex_;
  TensorMsg::TensorListMsg::ConstSharedPtr latest_pose_input_;

  rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr detection_pub_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr reset_client_;
  rclcpp::Service<isaac_ros_foundationpose::srv::SwitchMesh>::SharedPtr switch_mesh_srv_;
  rclcpp::Publisher<TensorMsg::TensorListMsg>::SharedPtr pose_matrix_pub_;

  // TRT topic pub/sub (blocking pattern)
  rclcpp::Publisher<TensorMsg::TensorListMsg>::SharedPtr refine_pub_;
  rclcpp::Subscription<TensorMsg::TensorListMsg>::SharedPtr refine_sub_;
  rclcpp::CallbackGroup::SharedPtr trt_callback_group_;

  // Blocking sync state for refine TRT
  std::mutex refine_mutex_;
  std::condition_variable refine_cv_;
  bool refine_result_ready_{false};
  TensorMsg::TensorListMsg refine_result_;

  // Persistent worker thread
  void workerLoop();
  std::thread worker_thread_;
  std::mutex work_mutex_;
  std::condition_variable work_cv_;
  std::function<void()> pending_work_;
  bool work_ready_{false};
  bool shutdown_{false};

  // CUDA stream
  cudaStream_t cuda_stream_{nullptr};

  // Pre-allocated GPU buffers (no cudaMalloc/cudaFree in callbacks)
  float * pose_input_gpu_{nullptr};        // [1,4,4] input pose from detection
  float * pc_gpu_{nullptr};                // [rgb_h * rgb_w * 3] lazy-alloc on first frame
  ImageDim pc_size_;

  int * support_count_device_{nullptr};    // single int on GPU for support-count kernel output

  // Pipeline components
  std::unique_ptr<MeshLoader> mesh_loader_;
  std::unique_ptr<PoseRenderer> renderer_;
  std::unique_ptr<PoseTransformer> transformer_;
  std::unique_ptr<PoseDecoder> decoder_;

  // Parameters
  std::string configuration_file_;
  std::string mesh_file_path_;
  StringList refine_input_tensor_names_;
  float min_depth_{0.1f};
  float max_depth_{2.0f};
  float rot_normalizer_{1.0f};
  uint32_t resized_image_width_{160};
  uint32_t resized_image_height_{160};
  float refine_crop_ratio_{1.2f};

  // Minimum number of point-cloud points that must lie within mesh_diameter/2 of
  // the predicted pose center for the pose to be considered valid (drift guard).
  int min_pointcloud_support_{50};
  bool enable_auto_reset_{true};

  std::string tf_frame_name_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::mutex mesh_mutex_;

  // Mesh switching is coordinated through the internal tracking/switch_mesh service.
};

}  // namespace foundationpose
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_FOUNDATIONPOSE__FOUNDATIONPOSE_TRACKING_NODE_HPP_
