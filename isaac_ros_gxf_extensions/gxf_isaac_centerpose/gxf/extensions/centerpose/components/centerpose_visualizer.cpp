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
#include "extensions/centerpose/components/centerpose_visualizer.hpp"

#include <utility>
#include <vector>

#include "cuda.h"          // NOLINT
#include "cuda_runtime.h"  // NOLINT
#include "Eigen/Dense"
#include "extensions/centerpose/components/centerpose_types.hpp"
#include "extensions/centerpose/components/cuboid3d.hpp"
#include "extensions/centerpose/components/video_buffer_utils.hpp"
#include "detection3_d_array_message.hpp"
#include "gxf/multimedia/camera.hpp"
#include "gxf/multimedia/video.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/opencv.hpp"

namespace nvidia {
namespace isaac {
namespace centerpose {

namespace {

gxf::Expected<void> CopyVideoBuffer(
    gxf::Handle<gxf::VideoBuffer> output_video_buffer,
    gxf::Handle<gxf::VideoBuffer> input_video_buffer,
    const cudaStream_t cuda_stream_) {
  cudaMemcpyKind operation;
  switch (input_video_buffer->storage_type()) {
    case gxf::MemoryStorageType::kDevice: {
      operation = cudaMemcpyDeviceToHost;
    } break;
    case gxf::MemoryStorageType::kHost:
    case gxf::MemoryStorageType::kSystem: {
      operation = cudaMemcpyHostToHost;
    } break;
    default: {
      return gxf::Unexpected{GXF_PARAMETER_OUT_OF_RANGE};
    } break;
  }
  // Add cuda stream here aboveand make it an async function call
  cudaError_t error = cudaMemcpyAsync(
      output_video_buffer->pointer(), input_video_buffer->pointer(), input_video_buffer->size(),
      operation, cuda_stream_);
  if (error != cudaSuccess) {
    GXF_LOG_ERROR("%s: %s", cudaGetErrorName(error), cudaGetErrorString(error));
    return gxf::Unexpected{GXF_FAILURE};
  }

  error = cudaStreamSynchronize(cuda_stream_);
  if (error != cudaSuccess) {
    GXF_LOG_ERROR("%s: %s", cudaGetErrorName(error), cudaGetErrorString(error));
    return gxf::Unexpected{GXF_FAILURE};
  }

  return gxf::Success;
}

Eigen::MatrixXfRM Calculate3DPoints(const Eigen::Matrix4f& pose_pred, const Cuboid3d& cuboid3d) {
  const std::array<
      Eigen::Vector3f, static_cast<size_t>(CuboidVertexType::TOTAL_CORNER_VERTEX_COUNT)>&
      points_3d_obj = cuboid3d.vertices();
  Eigen::MatrixXf points_3d_obj_stacked =
      Eigen::MatrixXf(4, static_cast<size_t>(CuboidVertexType::TOTAL_CORNER_VERTEX_COUNT));
  for (int i = 0; i < points_3d_obj_stacked.cols(); ++i) {
    points_3d_obj_stacked.block<3, 1>(0, i) = points_3d_obj[i];
    points_3d_obj_stacked(3, i) = 1.0f;
  }
  Eigen::MatrixXfRM points_3d_cam =
      (pose_pred * points_3d_obj_stacked).block(0, 0, 3, points_3d_obj_stacked.cols()).transpose();
  return points_3d_cam;
}

void DarkenROI(const std::vector<cv::Point2f>& reprojected_points, cv::Mat& img) {
  cv::Mat dark_layer = img.clone();

  std::vector<cv::Point2i> polys;
  for (const auto& point : reprojected_points) {
    polys.push_back(point);
  }

  std::vector<cv::Point2i> convex_hull(polys.size());
  cv::convexHull(polys, convex_hull);
  cv::fillConvexPoly(dark_layer, convex_hull, cv::Scalar{0.0, 0.0, 0.0});

  constexpr double kAlpha{0.7};
  cv::addWeighted(img, kAlpha, dark_layer, 1.0 - kAlpha, 0, img);
}

void DrawBoundingBox(
    const std::vector<cv::Point2f>& reprojected_points, const int32_t color_int, cv::Mat& img) {
  const std::vector<cv::Point2i> edges = {cv::Point2i{2, 4}, cv::Point2i{2, 6}, cv::Point2i{6, 8},
                                          cv::Point2i{4, 8}, cv::Point2i{1, 2}, cv::Point2i{3, 4},
                                          cv::Point2i{5, 6}, cv::Point2i{7, 8}, cv::Point2i{1, 3},
                                          cv::Point2i{1, 5}, cv::Point2i{3, 7}, cv::Point2i{5, 7}};

  cv::Scalar color = {
      static_cast<double>((color_int >> 16) & 0xFF), static_cast<double>((color_int >> 8) & 0xFF),
      static_cast<double>(color_int & 0xFF)};
  for (const cv::Point2i& edge : edges) {
    const int start_idx{edge.x - 1};
    const int end_idx{edge.y - 1};
    cv::line(img, reprojected_points[start_idx], reprojected_points[end_idx], color, 2);
  }
}

void DrawAxes(
    const Eigen::MatrixXfRM& keypoints3d, const Eigen::Matrix3f& camera_matrix, cv::Mat& img) {
  const std::vector<Eigen::Vector3f> axes_point_list = {
      Eigen::Vector3f{0.0f, 0.0f, 0.0f},
      keypoints3d.block<1, 3>(3, 0) - keypoints3d.block<1, 3>(1, 0),
      keypoints3d.block<1, 3>(2, 0) - keypoints3d.block<1, 3>(1, 0),
      keypoints3d.block<1, 3>(5, 0) - keypoints3d.block<1, 3>(1, 0),
  };

  std::vector<cv::Point2i> viewport_points;
  for (const auto& axes_point : axes_point_list) {
    Eigen::Vector3f vector = axes_point.norm() == 0.0f ? Eigen::Vector3f{0.0f, 0.0f, 0.0f}
                                                       : axes_point / axes_point.norm() * 0.5f;
    vector += keypoints3d.block<1, 3>(0, 0);
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

gxf::Expected<void> DrawDetections(
    Detection3DListMessageParts detections, gxf::Handle<gxf::VideoBuffer> output_video_buffer,
    Eigen::Matrix3f camera_matrix_eigen, const bool show_axes, const int32_t bounding_box_color) {
  cv::Mat img{
      static_cast<int32_t>(output_video_buffer->video_frame_info().height),
      static_cast<int32_t>(output_video_buffer->video_frame_info().width), CV_8UC3,
      output_video_buffer->pointer()};
  cv::Mat camera_matrix_cv;
  cv::eigen2cv(camera_matrix_eigen, camera_matrix_cv);
  for (size_t i = 0; i < detections.count; ++i) {
    gxf::Handle<Pose3d> pose = detections.poses.at(i).value();
    gxf::Handle<Vector3f> bbox_size = detections.bbox_sizes.at(i).value();

    Cuboid3d cuboid3d{*bbox_size};

    Eigen::MatrixXfRM points_3d_cam = Calculate3DPoints(pose->matrix().cast<float>(), cuboid3d);

    Eigen::Matrix3f rot_matrix_eigen = pose->rotation.matrix().cast<float>();
    cv::Mat rot_matrix;
    cv::eigen2cv(rot_matrix_eigen, rot_matrix);

    // Note: points3d is already given in camera frame
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);
    std::vector<cv::Point2f> reprojected_points;

    cv::Mat points_3d_cam_cv;
    cv::eigen2cv(points_3d_cam, points_3d_cam_cv);
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
  return gxf::Success;
}

gxf::Expected<Eigen::Matrix3f> GetCameraMatrix(gxf::Handle<gxf::CameraModel> camera_model) {
  if (!camera_model) {
    return gxf::Unexpected{GXF_FAILURE};
  }

  Eigen::Matrix3f camera_matrix = Eigen::Matrix3f::Identity();
  camera_matrix(0, 0) = camera_model->focal_length.x;
  camera_matrix(0, 2) = camera_model->principal_point.x;

  camera_matrix(1, 1) = camera_model->focal_length.y;
  camera_matrix(1, 2) = camera_model->principal_point.y;

  camera_matrix(2, 2) = 1.0f;
  return camera_matrix;
}

}  // namespace

gxf_result_t CenterPoseVisualizer::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(video_buffer_input_, "video_buffer_input");
  result &= registrar->parameter(detections_input_, "detections_input");
  result &= registrar->parameter(camera_model_input_, "camera_model_input");
  result &= registrar->parameter(output_, "output");
  result &= registrar->parameter(allocator_, "allocator");
  result &= registrar->parameter(show_axes_, "show_axes");
  result &= registrar->parameter(bounding_box_color_, "bounding_box_color");
  result &= registrar->parameter(
      cuda_stream_pool_, "stream_pool", "Cuda Stream Pool",
      "Instance of gxf::CudaStreamPool to allocate CUDA stream.");
  return gxf::ToResultCode(result);
}

gxf_result_t CenterPoseVisualizer::start() {
  // Get cuda stream from stream pool
  auto maybe_stream = cuda_stream_pool_.get()->allocateStream();
  if (!maybe_stream) {
    return gxf::ToResultCode(maybe_stream);
  }

  cuda_stream_handle_ = std::move(maybe_stream.value());
  if (!cuda_stream_handle_->stream()) {
    GXF_LOG_ERROR("Allocated stream is not initialized!");
    return GXF_FAILURE;
  }
  if (!cuda_stream_handle_.is_null()) {
    cuda_stream_ = cuda_stream_handle_->stream().value();
  }
  return GXF_SUCCESS;
}

gxf_result_t CenterPoseVisualizer::tick() {
  gxf::Entity video_buffer_input_entity, detections_input_entity, camera_model_input_entity;
  gxf::Handle<gxf::VideoBuffer> input_video_buffer;
  Detection3DListMessageParts detections;
  gxf::Handle<gxf::CameraModel> camera_model;

  Eigen::Matrix3f camera_matrix;
  auto maybe_received_inputs =
      video_buffer_input_->receive()
          .assign_to(video_buffer_input_entity)
          .and_then([&]() { return video_buffer_input_entity.get<gxf::VideoBuffer>(); })
          .assign_to(input_video_buffer)
          .and_then([&]() { return detections_input_->receive(); })
          .assign_to(detections_input_entity)
          .and_then([&]() { return GetDetection3DListMessage(detections_input_entity); })
          .assign_to(detections)
          .and_then([&]() { return camera_model_input_->receive(); })
          .assign_to(camera_model_input_entity)
          .and_then([&]() { return camera_model_input_entity.get<gxf::CameraModel>("intrinsics"); })
          .assign_to(camera_model)
          .and_then([&]() { return GetCameraMatrix(camera_model); })
          .assign_to(camera_matrix);
  if (!maybe_received_inputs) {
    GXF_LOG_ERROR("Failed to get all required inputs!");
    return gxf::ToResultCode(maybe_received_inputs);
  }

  gxf::Entity output_entity;
  gxf::Handle<gxf::VideoBuffer> output_video_buffer;
  auto maybe_created_outputs =
      gxf::Entity::New(context())
          .assign_to(output_entity)
          .and_then([&]() { return output_entity.add<gxf::VideoBuffer>(); })
          .assign_to(output_video_buffer)
          .and_then([&]() -> gxf::Expected<void> {
            switch (input_video_buffer->video_frame_info().color_format) {
              case gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB:
                return AllocateVideoBuffer<gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB>(
                    output_video_buffer, input_video_buffer->video_frame_info().width,
                    input_video_buffer->video_frame_info().height, gxf::MemoryStorageType::kHost,
                    allocator_.get());
              case gxf::VideoFormat::GXF_VIDEO_FORMAT_BGR:
                return AllocateVideoBuffer<gxf::VideoFormat::GXF_VIDEO_FORMAT_BGR>(
                    output_video_buffer, input_video_buffer->video_frame_info().width,
                    input_video_buffer->video_frame_info().height, gxf::MemoryStorageType::kHost,
                    allocator_.get());
              default: {
                GXF_LOG_ERROR("Received unsupported color format!");
                return gxf::Unexpected{GXF_FAILURE};
              }
            }
          });
  if (!maybe_created_outputs) {
    GXF_LOG_ERROR("Failed to create output video buffer!");
    return gxf::ToResultCode(maybe_created_outputs);
  }

  auto maybe_copied = CopyVideoBuffer(output_video_buffer, input_video_buffer, cuda_stream_);
  if (!maybe_copied) {
    return gxf::ToResultCode(maybe_copied);
  }
  DrawDetections(
      detections, output_video_buffer, camera_matrix, show_axes_.get(), bounding_box_color_.get());

  gxf::Handle<gxf::Timestamp> timestamp;
  auto maybe_added_timestamp =
      output_entity.add<gxf::Timestamp>("timestamp").assign_to(timestamp).and_then([&]() {
        *timestamp = *detections.timestamp;
      });
  if (!maybe_added_timestamp) {
    GXF_LOG_ERROR("Could not add timestamp!");
    return gxf::ToResultCode(maybe_added_timestamp);
  }

  return gxf::ToResultCode(output_->publish(output_entity));
}

gxf_result_t CenterPoseVisualizer::stop() {
  return GXF_SUCCESS;
}

}  // namespace centerpose
}  // namespace isaac
}  // namespace nvidia
