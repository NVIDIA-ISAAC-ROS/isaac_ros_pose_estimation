// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_centerpose/centerpose_visualizer_node.hpp"

#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "isaac_ros_centerpose/cuboid3d.hpp"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include "vision_msgs/msg/detection3_d.hpp"
#include "vision_msgs/msg/object_hypothesis_with_pose.hpp"
#include "vision_msgs/msg/bounding_box3_d.hpp"

#include "rclcpp/rclcpp.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/opencv.hpp"


namespace nvidia
{
namespace isaac_ros
{
namespace centerpose
{

namespace
{

constexpr char INPUT_IMAGE_TOPIC_NAME[] = "image";
constexpr char INPUT_DETECTION_TOPIC_NAME[] = "centerpose/detections";
constexpr char INPUT_CAMERA_INFO_TOPIC_NAME[] = "camera_info";
constexpr char OUTPUT_TOPIC_NAME[] = "centerpose/image_visualized";

Eigen::Matrix3f GetCameraMatrix(
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info)
{
  Eigen::Matrix3f camera_matrix = Eigen::Matrix3f::Identity();
  camera_matrix(0, 0) = camera_info->k[0];
  camera_matrix(0, 2) = camera_info->k[2];
  camera_matrix(1, 1) = camera_info->k[4];
  camera_matrix(1, 2) = camera_info->k[5];
  return camera_matrix;
}

Eigen::MatrixXfRM Calculate3DPoints(
  const Eigen::Matrix4f & pose_pred, const Cuboid3d & cuboid3d)
{
  const std::array<
    Eigen::Vector3f, static_cast<size_t>(CuboidVertexType::TOTAL_CORNER_VERTEX_COUNT)> &
  points_3d_obj = cuboid3d.vertices();
  Eigen::MatrixXf points_3d_obj_stacked =
    Eigen::MatrixXf(4, static_cast<size_t>(CuboidVertexType::TOTAL_CORNER_VERTEX_COUNT));
  for (int i = 0; i < points_3d_obj_stacked.cols(); ++i) {
    points_3d_obj_stacked.block<3, 1>(0, i) = points_3d_obj[i];
    points_3d_obj_stacked(3, i) = 1.0f;
  }
  return (pose_pred * points_3d_obj_stacked)
         .block(0, 0, 3, points_3d_obj_stacked.cols()).transpose();
}

void DarkenROI(const std::vector<cv::Point2f> & reprojected_points, cv::Mat & img)
{
  if (reprojected_points.size() < 3) {
    return;
  }

  cv::Mat dark_layer = img.clone();

  std::vector<cv::Point2i> polys;
  polys.reserve(reprojected_points.size());
  for (const auto & point : reprojected_points) {
    polys.push_back(point);
  }

  std::vector<cv::Point2i> convex_hull(polys.size());
  cv::convexHull(polys, convex_hull);
  cv::fillConvexPoly(dark_layer, convex_hull, cv::Scalar{0.0, 0.0, 0.0});

  constexpr double kAlpha{0.7};
  cv::addWeighted(img, kAlpha, dark_layer, 1.0 - kAlpha, 0, img);
}

void DrawBoundingBox(
  const std::vector<cv::Point2f> & reprojected_points, const int32_t color_int, cv::Mat & img)
{
  if (reprojected_points.size() < static_cast<size_t>(
      CuboidVertexType::TOTAL_CORNER_VERTEX_COUNT))
  {
    return;
  }

  const std::array<std::pair<int32_t, int32_t>, 12> edges = {{
    {1, 3}, {1, 5}, {5, 7}, {3, 7},
    {0, 1}, {2, 3}, {4, 5}, {6, 7},
    {0, 2}, {0, 4}, {2, 6}, {4, 6}}};

  const cv::Scalar color = {
    static_cast<double>((color_int >> 16) & 0xFF),
    static_cast<double>((color_int >> 8) & 0xFF),
    static_cast<double>(color_int & 0xFF)};

  for (const auto & edge : edges) {
    cv::line(img, reprojected_points[edge.first], reprojected_points[edge.second], color, 2);
  }
}

void DrawAxes(
  const Eigen::MatrixXfRM & keypoints3d, const Eigen::Matrix3f & camera_matrix, cv::Mat & img)
{
  const std::vector<Eigen::Vector3f> axes_point_list = {
    Eigen::Vector3f{0.0f, 0.0f, 0.0f},
    keypoints3d.block<1, 3>(3, 0) - keypoints3d.block<1, 3>(1, 0),
    keypoints3d.block<1, 3>(2, 0) - keypoints3d.block<1, 3>(1, 0),
    keypoints3d.block<1, 3>(5, 0) - keypoints3d.block<1, 3>(1, 0),
  };

  std::vector<cv::Point2i> viewport_points;
  viewport_points.reserve(axes_point_list.size());
  for (const auto & axes_point : axes_point_list) {
    Eigen::Vector3f vector = axes_point.norm() == 0.0f ?
      Eigen::Vector3f{0.0f, 0.0f, 0.0f} :
    axes_point / axes_point.norm() * 0.5f;
    vector += keypoints3d.block<1, 3>(0, 0).transpose();
    Eigen::Vector3f pp = camera_matrix * vector;
    if (pp.z() != 0.0f) {
      pp.x() = pp.x() / pp.z();
      pp.y() = pp.y() / pp.z();
    }
    viewport_points.push_back(cv::Point2i{static_cast<int>(pp.x()), static_cast<int>(pp.y())});
  }

  const std::array<cv::Scalar, 3> colors = {
    cv::Scalar{0, 255, 0}, cv::Scalar{255, 0, 0}, cv::Scalar{0, 0, 255}};

  for (size_t i = 0; i < colors.size(); ++i) {
    cv::line(img, viewport_points[0], viewport_points[i + 1], colors[i], 5);
  }
}

void DrawDetections(
  const vision_msgs::msg::Detection3DArray::ConstSharedPtr & detections,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info,
  const bool show_axes, const int32_t bounding_box_color, cv::Mat & img)
{
  const Eigen::Matrix3f camera_matrix_eigen = GetCameraMatrix(camera_info);
  cv::Mat camera_matrix_cv;
  cv::eigen2cv(camera_matrix_eigen, camera_matrix_cv);

  for (const auto & detection : detections->detections) {
    const Eigen::Vector3f bbox_size{
      static_cast<float>(detection.bbox.size.x),
      static_cast<float>(detection.bbox.size.y),
      static_cast<float>(detection.bbox.size.z)};
    Cuboid3d cuboid3d{bbox_size};

    const auto & pose = detection.bbox.center;
    Eigen::Quaternionf orientation{
      static_cast<float>(pose.orientation.w),
      static_cast<float>(pose.orientation.x),
      static_cast<float>(pose.orientation.y),
      static_cast<float>(pose.orientation.z)};
    if (orientation.norm() == 0.0f) {
      continue;
    }

    Eigen::Matrix4f pose_pred = Eigen::Matrix4f::Identity();
    pose_pred.block<3, 3>(0, 0) = orientation.normalized().toRotationMatrix();
    pose_pred.block<3, 1>(0, 3) = Eigen::Vector3f{
      static_cast<float>(pose.position.x),
      static_cast<float>(pose.position.y),
      static_cast<float>(pose.position.z)};

    Eigen::MatrixXfRM points_3d_cam = Calculate3DPoints(pose_pred, cuboid3d);

    cv::Mat points_3d_cam_cv;
    cv::eigen2cv(points_3d_cam, points_3d_cam_cv);
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);
    std::vector<cv::Point2f> reprojected_points;
    cv::projectPoints(
      points_3d_cam_cv, rvec, tvec, camera_matrix_cv, dist_coeffs, reprojected_points);

    DarkenROI(reprojected_points, img);
    DrawBoundingBox(reprojected_points, bounding_box_color, img);

    if (show_axes) {
      Eigen::Vector3f points_3d_cam_mean = points_3d_cam.colwise().mean().transpose();
      Eigen::MatrixXfRM keypoints3d(1 + points_3d_cam.rows(), points_3d_cam.cols());
      keypoints3d << points_3d_cam_mean, points_3d_cam;
      DrawAxes(keypoints3d, camera_matrix_eigen, img);
    }
  }
}

}  // namespace


CenterPoseVisualizerNode::CenterPoseVisualizerNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("centerpose_visualizer", options),
  show_axes_{declare_parameter<bool>("show_axes", true)},
  bounding_box_color_{static_cast<int32_t>(declare_parameter<int32_t>(
      "bounding_box_color", int32_t{0x000000ff}))},
  memory_pool_block_size_{declare_parameter<int64_t>(
    "memory_pool_block_size", static_cast<int64_t>(3) * 1024 * 1024 * 4)},
  memory_pool_num_blocks_{declare_parameter<int64_t>(
    "memory_pool_num_blocks", int64_t{40})},
  input_queue_size_{static_cast<int16_t>(declare_parameter<int>(
    "input_queue_size", 10))},
  output_queue_size_{static_cast<int16_t>(declare_parameter<int>(
    "output_queue_size", 10))},
  image_sub_{},
  detection3darray_sub_{},
  camera_info_sub_{},
  image_camera_info_sync_{ExactPolicy(static_cast<uint32_t>(input_queue_size_)), image_sub_,
    detection3darray_sub_, camera_info_sub_}
{
  // Create CUDA resources
  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("CenterPoseVisualizerNode");
  CHECK_CUDA_ERROR(pool_.create(
    static_cast<size_t>(memory_pool_block_size_),
    static_cast<size_t>(memory_pool_num_blocks_),
    nvidia::isaac_ros::nitros::CUDAMemoryPool::MemoryType::Device),
    "Failed to create CUDA memory pool");

  // This function sets the QoS parameter for publishers and subscribers setup by this NITROS node
  const rclcpp::QoS input_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "input_qos").keep_last(input_queue_size_);
  const rclcpp::QoS output_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "output_qos").keep_last(output_queue_size_);
  const rmw_qos_profile_t rmw_qos_profile = input_qos.get_rmw_qos_profile();

  // Subscription options (can be used for callback groups, etc.)
  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  // Publisher options
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  // Create subscribers
  image_camera_info_sync_.registerCallback(
    std::bind(
      &CenterPoseVisualizerNode::InputCallback, this,
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
  image_sub_.subscribe(this, INPUT_IMAGE_TOPIC_NAME, rmw_qos_profile, sub_options);
  camera_info_sub_.subscribe(this, INPUT_CAMERA_INFO_TOPIC_NAME, rmw_qos_profile, sub_options);
  detection3darray_sub_.subscribe(this, INPUT_DETECTION_TOPIC_NAME, rmw_qos_profile, sub_options);

  image_pub_ = create_publisher<nvidia::isaac_ros::nitros::NitrosImage>(
    OUTPUT_TOPIC_NAME, output_qos, pub_options);

  RCLCPP_INFO(get_logger(), "CenterPoseVisualizerNode initialized");
}

void CenterPoseVisualizerNode::InputCallback(
  const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & nitros_image,
  const vision_msgs::msg::Detection3DArray::ConstSharedPtr & detection3darray,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info
)
{
  RCLCPP_DEBUG(get_logger(), "CenterPoseVisualizerNode input callback");

  const cudaStream_t stream = *cuda_stream_;

  // Device -> host for OpenCV drawing
  cv::Mat image_mat;
  {
    const int cv_type = CV_8UC3;
    std::vector<uint8_t> host_image_buffer(
      static_cast<size_t>(nitros_image->step) * static_cast<size_t>(nitros_image->height));

    cudaError_t cuda_result = cudaMemcpyAsync(
      host_image_buffer.data(),
      nitros_image->get_read_handle(stream).get_ptr(),
      nitros_image->step * nitros_image->height,
      cudaMemcpyDeviceToHost, stream);
    CHECK_CUDA_ERROR(cuda_result, "Failed to copy image data from device to host");

    cuda_result = cudaStreamSynchronize(stream);
    CHECK_CUDA_ERROR(cuda_result, "Failed to synchronize CUDA stream");

    image_mat = cv::Mat(
      static_cast<int>(nitros_image->height),
      static_cast<int>(nitros_image->width),
      cv_type,
      host_image_buffer.data(),
      static_cast<size_t>(nitros_image->step)).clone();
  }

  DrawDetections(detection3darray, camera_info, show_axes_, bounding_box_color_, image_mat);

  // Host -> device: publish annotated image in a pool-backed NitrosImage
  auto output_image = std::make_unique<nvidia::isaac_ros::nitros::NitrosImage>();
  auto output_write_handle = output_image->from_pool(
    pool_,
    nitros_image->width,
    nitros_image->height,
    nitros_image->step,
    nitros_image->encoding,
    stream);

  const size_t image_bytes =
    static_cast<size_t>(nitros_image->step) * static_cast<size_t>(nitros_image->height);
  cudaError_t cuda_result = cudaMemcpyAsync(
    output_write_handle.get_ptr(),
    image_mat.data,
    image_bytes,
    cudaMemcpyHostToDevice,
    stream);
  CHECK_CUDA_ERROR(cuda_result, "Failed to copy annotated image to device buffer");

  output_image->frame_id = nitros_image->get_frame_id();
  output_image->set_timestamp_sec(nitros_image->get_timestamp_sec());
  output_image->set_timestamp_nsec(nitros_image->get_timestamp_nsec());
  output_image->data_format_name = nitros_image->data_format_name;
  output_image->compatible_data_format_name = nitros_image->compatible_data_format_name;

  image_pub_->publish(*output_image);
}

}  // namespace centerpose
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::centerpose::CenterPoseVisualizerNode)
