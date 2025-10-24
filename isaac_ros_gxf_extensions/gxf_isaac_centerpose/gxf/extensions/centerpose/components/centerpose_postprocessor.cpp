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
#include "extensions/centerpose/components/centerpose_postprocessor.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cuda.h"          // NOLINT
#include "cuda_runtime.h"  // NOLINT
#include "Eigen/Dense"
#include "extensions/centerpose/components/centerpose_detection.hpp"
#include "extensions/centerpose/components/cuboid3d.hpp"
#include "extensions/centerpose/components/cuboid_pnp_solver.hpp"
#include "extensions/centerpose/components/soft_nms_nvidia.hpp"
#include "gxf/multimedia/camera.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/timestamp.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/opencv.hpp"

namespace nvidia {
namespace isaac {
namespace centerpose {

namespace {

constexpr std::array<const char*, 7> kTensorIdxToStr = {
    "bboxes", "scores", "kps", "clses", "obj_scale", "kps_displacement_mean", "kps_heatmap_mean"};

const std::unordered_map<const char*, int> kTensorStrToIdx = {
    {"bboxes", 0},          {"scores", 1},    {"kps", 2},
    {"clses", 3},           {"obj_scale", 4}, {"kps_displacement_mean", 5},
    {"kps_heatmap_mean", 6}};

constexpr float kNMSNt{0.5f};
constexpr float kNMSSigma{0.5f};
const NMSMethod kNMSMethod{NMSMethod::GAUSSIAN};

constexpr char kTimestampName[] = "timestamp";

inline gxf::Expected<std::vector<gxf::Handle<gxf::Tensor>>> GetInputTensors(
    const gxf::Entity entity) {
  std::vector<gxf::Handle<gxf::Tensor>> tensors;
  tensors.reserve(kTensorIdxToStr.size());
  for (const auto& tensor_name : kTensorIdxToStr) {
    auto maybe_tensor = entity.get<gxf::Tensor>(tensor_name);
    if (!maybe_tensor) {
      GXF_LOG_ERROR("Failed to get tensor element: %s", tensor_name);
      return gxf::Unexpected{GXF_FAILURE};
    }
    tensors.push_back(maybe_tensor.value());
  }
  return tensors;
}

Eigen::Matrix3fRM ComputeAffineTransform(
    const Eigen::Vector2f& eigen_center, const float scale_scalar, const float rot,
    const Eigen::Vector2i& output_field_size, bool inv = false) {
  const cv::Point2f shift = {0.0f, 0.0f};

  const float src_w{scale_scalar};
  const float dst_w{static_cast<float>(output_field_size(0))};
  const float dst_h{static_cast<float>(output_field_size(1))};

  const float rot_rad{static_cast<float>(M_PI * rot / 180.0f)};
  auto calculate_direction = [](const cv::Point2f& pt, const float rot_rad) {
    return cv::Point2f{
        pt.x * std::cos(rot_rad) - pt.y * std::sin(rot_rad),
        pt.x * std::sin(rot_rad) + pt.y * std::cos(rot_rad)};
  };
  const cv::Point2f src_direction = calculate_direction({0.0f, src_w * -0.5f}, rot_rad);
  const cv::Point2f dst_direction = cv::Point2f{0.0f, dst_w * -0.5f};

  const cv::Point2f center = {eigen_center(0), eigen_center(1)};

  // Compute the points of interest in the original image
  std::vector<cv::Point2f> src_points;
  src_points.push_back(center + scale_scalar * shift);
  src_points.push_back(center + src_direction + scale_scalar * shift);

  auto calculate_third_point = [](const cv::Point2f& a, const cv::Point2f& b) {
    const cv::Point2f direction = a - b;
    return b + cv::Point2f{-direction.y, direction.x};
  };
  src_points.push_back(calculate_third_point(src_points[0], src_points[1]));

  // Compute the corresponding points of interest in the output_field_size image plane
  std::vector<cv::Point2f> dst_points;
  dst_points.push_back({dst_w * 0.5f, dst_h * 0.5f});
  dst_points.push_back(cv::Point2f(dst_w * 0.5f, dst_h * 0.5f) + dst_direction);
  dst_points.push_back(calculate_third_point(dst_points[0], dst_points[1]));

  // Use the computed src and dst points to find the affine transform (mapping
  // b/w the two)
  cv::Mat affine_matrix_cv = inv ? cv::getAffineTransform(dst_points, src_points)
                                 : cv::getAffineTransform(src_points, dst_points);
  Eigen::Matrix3dRM affine_matrix = Eigen::Matrix3dRM::Identity();
  cv::cv2eigen(affine_matrix_cv, affine_matrix);

  return affine_matrix.cast<float>();
}

template <typename T, int rows, int cols, int type>
gxf::Expected<void> ToTensor(
    const std::vector<Eigen::Matrix<T, rows, cols, type>>& eigen_type,
    gxf::Handle<gxf::Tensor> tensor, gxf::Handle<gxf::Allocator> allocator,
    const gxf::MemoryStorageType storage_type, const cudaMemcpyKind operation,
    const cudaStream_t cuda_stream_) {
  // Handle special case when input is empty
  if (eigen_type.size() == 0) {
    return gxf::Success;
  }
  return tensor
      ->reshape<T>(
          gxf::Shape{
              static_cast<int>(eigen_type.size()), static_cast<int>(eigen_type[0].rows()),
              static_cast<int>(eigen_type[0].cols())},
          storage_type, allocator)
      .and_then([&]() -> gxf::Expected<void> {
        for (size_t i = 0; i < eigen_type.size(); ++i) {
          size_t size = tensor->size() / tensor->shape().dimension(0);
          size_t start = i * size;
          // Add cuda stream here aboveand make it an async function call
          cudaError_t error = cudaMemcpyAsync(
              tensor->pointer() + start, eigen_type[i].data(), size, operation, cuda_stream_);
          if (error != cudaSuccess) {
            GXF_LOG_ERROR("Error while copying from device to host: %s", cudaGetErrorString(error));
            return gxf::Unexpected{GXF_FAILURE};
          }
        }
        return gxf::Success;
      });
}

template <typename T>
gxf::Expected<void> ToTensor(
    const std::vector<T>& vec, gxf::Handle<gxf::Tensor> tensor,
    gxf::Handle<gxf::Allocator> allocator, const gxf::MemoryStorageType storage_type,
    const cudaMemcpyKind operation, const cudaStream_t cuda_stream_) {
  // handle when input is empty
  if (vec.size() == 0) {
    return gxf::Success;
  }
  return tensor->reshape<T>(gxf::Shape{static_cast<int>(vec.size()), 1}, storage_type, allocator)
      .and_then([&]() -> gxf::Expected<void> {
        // Add cuda stream here aboveand make it an async function call
        const cudaError_t error =
            cudaMemcpyAsync(tensor->pointer(), vec.data(), tensor->size(), operation,
                cuda_stream_);
        if (error != cudaSuccess) {
          GXF_LOG_ERROR("Error while copying from device to host: %s", cudaGetErrorString(error));
          return gxf::Unexpected{GXF_FAILURE};
        }
        return gxf::Success;
      });
}

inline Eigen::MatrixXfRM PerformAffineTransform(
    const Eigen::Ref<const Eigen::MatrixXfRM>& untransformed_points,
    const Eigen::Matrix3fRM& affine_transform) {
  Eigen::MatrixXfRM transformed_points =
      affine_transform.block<2, 2>(0, 0) * untransformed_points.transpose();
  transformed_points.colwise() += affine_transform.block<2, 1>(0, 2);
  return transformed_points.transpose();
}

inline Eigen::MatrixXfRM Calculate2DKeypoints(
    const Eigen::MatrixXfRM& kps_displacement_mean, const Eigen::Matrix3fRM& affine_transform) {
  constexpr int32_t kps_displacement_rows{8};
  constexpr int32_t kps_displacement_cols{2};
  Eigen::Map<const Eigen::MatrixXfRM> reshaped_kps_displacement_mean{
      kps_displacement_mean.data(), kps_displacement_rows, kps_displacement_cols};
  return PerformAffineTransform(reshaped_kps_displacement_mean, affine_transform);
}

inline Eigen::MatrixXfRM CalculateBBoxPoints(
    const Eigen::MatrixXfRM& bbox, const Eigen::Matrix3fRM& affine_transform) {
  constexpr int32_t bbox_rows{2};
  constexpr int32_t bbox_cols{2};
  Eigen::Map<const Eigen::MatrixXfRM> reshaped_bbox{bbox.data(), bbox_rows, bbox_cols};
  return PerformAffineTransform(reshaped_bbox, affine_transform);
}

gxf::Expected<PnPResult> SolvePnP(
    const Eigen::MatrixXfRM& keypoints2d, const Eigen::MatrixXfRM& kps_heatmap_mean,
    const Cuboid3d& cuboid3d, const Eigen::Matrix3f& camera_matrix) {
  constexpr int32_t kps_heatmap_rows{8};
  constexpr int32_t kps_heatmap_cols{2};
  Eigen::Map<const Eigen::MatrixXfRM> reshaped_kps_heatmap_mean(
      kps_heatmap_mean.data(), kps_heatmap_rows, kps_heatmap_cols);
  Eigen::MatrixXfRM points_filtered(
      keypoints2d.rows() + reshaped_kps_heatmap_mean.rows(), keypoints2d.cols());
  points_filtered << keypoints2d, reshaped_kps_heatmap_mean;
  Eigen::Vector4f dist_coeffs{0.0f, 0.0f, 0.0f, 0.0f};

  constexpr float pnp_scale_factor{1.0f};
  CuboidPnPSolver pnp_solver{pnp_scale_factor, camera_matrix, dist_coeffs, cuboid3d};
  auto maybe_result = pnp_solver.solvePnP(points_filtered);
  if (!maybe_result) {
    return gxf::Unexpected{GXF_FAILURE};
  }
  return maybe_result.value();
}

Eigen::MatrixXfRM Calculate3DPoints(const PnPResult& pnp_result, const Cuboid3d& cuboid3d) {
  Eigen::Matrix4f pose_pred = Eigen::Matrix4f::Identity();
  pose_pred.block<3, 3>(0, 0) = pnp_result.pose.orientation.normalized().toRotationMatrix();
  pose_pred.block<3, 1>(0, 3) = pnp_result.pose.position;

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

}  // namespace

gxf_result_t CenterPosePostProcessor::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(input_, "input", "Input", "The input");
  result &= registrar->parameter(camera_model_input_, "camera_model_input");
  result &= registrar->parameter(output_, "output", "Output", "The output");
  result &= registrar->parameter(allocator_, "allocator", "Allocator", "The allocator");
  result &= registrar->parameter(
      output_field_size_param_, "output_field_size", "Output Field Size",
      "The size of the 2D keypoint decoding from the network output");
  result &= registrar->parameter(
      cuboid_scaling_factor_, "cuboid_scaling_factor", "Cuboid Scaling Factor",
      "Used to scale the cuboid used for calculating "
      "the size of the objects detected",
      1.0f);
  result &= registrar->parameter(
      storage_type_, "storage type", "Storage Type", "Memory storage type of output frames.",
      static_cast<int32_t>(gxf::MemoryStorageType::kHost));
  result &= registrar->parameter(
      score_threshold_, "score_threshold", "Score Threshold",
      "Any detections with scores less than this value will be discarded");
  result &= registrar->parameter(
      cuda_stream_pool_, "stream_pool", "Cuda Stream Pool",
      "Instance of gxf::CudaStreamPool to allocate CUDA stream.");
  return gxf::ToResultCode(result);
}

gxf_result_t CenterPosePostProcessor::start() {
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
  camera_matrix_ = Eigen::Matrix3f::Identity();
  original_image_size_ = Eigen::Vector2i{0, 0};
  output_field_size_ =
      Eigen::Vector2i(output_field_size_param_.get()[0], output_field_size_param_.get()[1]);
  return GXF_SUCCESS;
}

gxf::Expected<void> CenterPosePostProcessor::updateCameraProperties(
    gxf::Handle<gxf::CameraModel> camera_model) {
  if (!camera_model) {
    return gxf::Unexpected{GXF_FAILURE};
  }

  camera_matrix_(0, 0) = camera_model->focal_length.x;
  camera_matrix_(0, 2) = camera_model->principal_point.x;

  camera_matrix_(1, 1) = camera_model->focal_length.y;
  camera_matrix_(1, 2) = camera_model->principal_point.y;

  camera_matrix_(2, 2) = 1.0f;

  // Avoid re-computing affine transform if it's not necessary
  if (original_image_size_.x() == static_cast<int32_t>(camera_model->dimensions.x) &&
      original_image_size_.y() == static_cast<int32_t>(camera_model->dimensions.y)) {
    return gxf::Success;
  }

  original_image_size_ = Eigen::Vector2i(camera_model->dimensions.x, camera_model->dimensions.y);

  const Eigen::Vector2f center = original_image_size_.cast<float>() / 2.0f;
  const float scale = std::max(
      static_cast<float>(camera_model->dimensions.x),
      static_cast<float>(camera_model->dimensions.y));

  constexpr float rotation_deg{0.0f};
  constexpr bool inverse{true};
  affine_transform_ =
      ComputeAffineTransform(center, scale, rotation_deg, output_field_size_, inverse);
  return gxf::Success;
}

gxf_result_t CenterPosePostProcessor::tick() {
  auto maybe_camera_model_entity = camera_model_input_->receive();
  if (!maybe_camera_model_entity) {
    GXF_LOG_ERROR("Failed to receive input camera model entity!");
    return gxf::ToResultCode(maybe_camera_model_entity);
  }

  auto maybe_camera_model = maybe_camera_model_entity.value().get<gxf::CameraModel>("intrinsics");
  if (!maybe_camera_model) {
    GXF_LOG_ERROR("Failed to receive input camera model!");
    return gxf::ToResultCode(maybe_camera_model);
  }

  auto maybe_updated_properties = updateCameraProperties(maybe_camera_model.value());
  if (!maybe_updated_properties) {
    GXF_LOG_ERROR("Failed to update camera properties!");
    return gxf::ToResultCode(maybe_updated_properties);
  }

  auto maybe_tensor_entity = input_->receive();
  if (!maybe_tensor_entity) {
    GXF_LOG_ERROR("Failed to receive input message!");
    return gxf::ToResultCode(maybe_tensor_entity);
  }
  gxf::Entity tensor_entity = maybe_tensor_entity.value();

  // Extract all tensors
  auto maybe_tensors = GetInputTensors(tensor_entity);
  if (!maybe_tensors) {
    return gxf::ToResultCode(maybe_tensors);
  }
  std::vector<gxf::Handle<gxf::Tensor>> tensors = maybe_tensors.value();

  // Assume all tensors share the same batch size
  CenterPoseDetectionList detections;
  for (int32_t batch = 0; batch < tensors[0]->shape().dimension(0); ++batch) {
    std::vector<Eigen::MatrixXfRM> batch_tensors;
    // Transfer from device to host
    for (size_t i = 0; i < tensors.size(); ++i) {
      batch_tensors.push_back(
          Eigen::MatrixXfRM(tensors[i]->shape().dimension(1), tensors[i]->shape().dimension(2)));

      size_t tensor_size = tensors[i]->size() / tensors[i]->shape().dimension(0);
      size_t tensor_start_idx = batch * tensor_size;
      // Add cuda stream here aboveand make it an async function call
      const cudaError_t memcpy_error = cudaMemcpyAsync(
          batch_tensors[i].data(), tensors[i]->pointer() + tensor_start_idx, tensor_size,
          cudaMemcpyDeviceToHost, cuda_stream_);
      if (memcpy_error != cudaSuccess) {
        GXF_LOG_ERROR("Error: %s", cudaGetErrorString(memcpy_error));
        return GXF_FAILURE;
      }
    }
    CenterPoseDetectionList batch_detections = processTensor(batch_tensors);
    detections.insert(detections.end(), batch_detections.begin(), batch_detections.end());
  }

  auto maybe_timestamp = tensor_entity.get<gxf::Timestamp>(kTimestampName);
  if (!maybe_timestamp) {
    GXF_LOG_ERROR("Failed to get timestamp!");
    return gxf::ToResultCode(maybe_timestamp);
  }

  return gxf::ToResultCode(publish(detections, maybe_timestamp.value()));
}

CenterPoseDetectionList CenterPosePostProcessor::processTensor(
    const std::vector<Eigen::MatrixXfRM>& tensors) {
  CenterPoseDetectionList detections;
  for (int i = 0; i < tensors[kTensorStrToIdx.at("scores")].rows(); ++i) {
    const float score{tensors[kTensorStrToIdx.at("scores")](i, 0)};
    const int cls{static_cast<int>(tensors[kTensorStrToIdx.at("clses")](i, 0))};
    if (score < score_threshold_.get()) {
      continue;
    }
    CenterPoseDetection detection;
    detection.class_id = cls;
    detection.score = score;
    constexpr int32_t keypoints_size_flattened{16};
    constexpr int32_t bbox_size_flattened{4};
    constexpr int32_t kps_heatmap_size_flattened{16};
    constexpr int32_t obj_scale_size_flattened{3};
    detection.keypoints2d = Calculate2DKeypoints(
        tensors[kTensorStrToIdx.at("kps_displacement_mean")].block<1, keypoints_size_flattened>(
            i, 0),
        affine_transform_);
    detection.bbox = CalculateBBoxPoints(
        tensors[kTensorStrToIdx.at("bboxes")].block<1, bbox_size_flattened>(i, 0),
        affine_transform_);
    detection.kps_heatmap_mean =
        tensors[kTensorStrToIdx.at("kps_heatmap_mean")].block<1, kps_heatmap_size_flattened>(i, 0);
    detection.bbox_size =
        cuboid_scaling_factor_.get() *
        tensors[kTensorStrToIdx.at("obj_scale")].block<1, obj_scale_size_flattened>(i, 0);
    detections.push_back(detection);
  }

  std::set<size_t> indices =
      SoftNMSNvidia(score_threshold_.get(), kNMSSigma, kNMSNt, kNMSMethod, &detections);

  CenterPoseDetectionList filtered_detections;
  for (const size_t& idx : indices) {
    filtered_detections.push_back(detections[idx]);
  }

  for (CenterPoseDetection& detection : filtered_detections) {
    Cuboid3d cuboid3d{detection.bbox_size};
    auto maybe_pnp_result =
        SolvePnP(detection.keypoints2d, detection.kps_heatmap_mean, cuboid3d, camera_matrix_);
    if (!maybe_pnp_result) {
      continue;
    }
    PnPResult pnp_result = maybe_pnp_result.value();
    pnp_result.pose.orientation.normalize();
    Eigen::MatrixXfRM points_3d_cam = Calculate3DPoints(pnp_result, cuboid3d);

    Eigen::Vector3f points_3d_cam_mean = points_3d_cam.colwise().mean().transpose();
    Eigen::Vector2f projected_points_mean =
        pnp_result.projected_points.colwise().mean().transpose();

    Eigen::MatrixXfRM keypoints3d(1 + points_3d_cam.rows(), points_3d_cam.cols());
    keypoints3d << points_3d_cam_mean, points_3d_cam;

    Eigen::MatrixXfRM projected_keypoints2d(
        1 + pnp_result.projected_points.rows(), pnp_result.projected_points.cols());
    projected_keypoints2d << projected_points_mean, pnp_result.projected_points;
    detection.projected_keypoints_2d = projected_keypoints2d;
    detection.keypoints3d = keypoints3d;
    detection.position = pnp_result.pose.position;
    detection.quaternion = pnp_result.pose.orientation;
  }

  return filtered_detections;
}

gxf::Expected<void> CenterPosePostProcessor::publish(
    const CenterPoseDetectionList& detections, gxf::Handle<gxf::Timestamp> input_timestamp) {
  std::vector<int64_t> class_id;
  std::vector<Eigen::MatrixXfRM> keypoints2d, projected_keypoints_2d, keypoints3d;
  std::vector<Eigen::Vector3f> position;
  std::vector<Eigen::Vector4f> quaternion;
  std::vector<float> score;
  std::vector<Eigen::Vector3f> bbox_size;

  for (const auto& detection : detections) {
    class_id.push_back(detection.class_id);
    keypoints2d.push_back(detection.keypoints2d);
    projected_keypoints_2d.push_back(detection.projected_keypoints_2d);
    keypoints3d.push_back(detection.keypoints3d);
    position.push_back(detection.position);
    quaternion.push_back(Eigen::Vector4f{
        detection.quaternion.x(), detection.quaternion.y(), detection.quaternion.z(),
        detection.quaternion.w()});
    score.push_back(detection.score);
    bbox_size.push_back(detection.bbox_size);
  }

  const gxf::MemoryStorageType storage_type{
      static_cast<gxf::MemoryStorageType>(storage_type_.get())};
  cudaMemcpyKind operation;
  switch (storage_type) {
    case gxf::MemoryStorageType::kDevice: {
      operation = cudaMemcpyHostToDevice;
    } break;
    case gxf::MemoryStorageType::kHost:
    case gxf::MemoryStorageType::kSystem: {
      operation = cudaMemcpyHostToHost;
    } break;
    default: {
      return gxf::Unexpected{GXF_PARAMETER_OUT_OF_RANGE};
    } break;
  }

  gxf::Entity output_entity;
  gxf::Handle<gxf::Tensor> class_id_tensor, keypoints2d_tensor, projected_keypoints_2d_tensor,
      keypoints3d_tensor, position_tensor, quaternion_tensor, score_tensor, bbox_size_tensor;
  gxf::Handle<gxf::Timestamp> output_timestamp;
  return gxf::Entity::New(context())
      .assign_to(output_entity)
      .and_then([&]() { return output_entity.add<gxf::Tensor>("class_id"); })
      .assign_to(class_id_tensor)
      .and_then([&]() {
        return ToTensor(class_id, class_id_tensor, allocator_.get(), storage_type,
            operation, cuda_stream_);
      })
      .and_then([&]() { return output_entity.add<gxf::Tensor>("keypoints2d"); })
      .assign_to(keypoints2d_tensor)
      .and_then([&]() {
        return ToTensor(keypoints2d, keypoints2d_tensor, allocator_.get(), storage_type, operation,
            cuda_stream_);
      })
      .and_then([&]() { return output_entity.add<gxf::Tensor>("projected_keypoints2d"); })
      .assign_to(projected_keypoints_2d_tensor)
      .and_then([&]() {
        return ToTensor(
            projected_keypoints_2d, projected_keypoints_2d_tensor, allocator_.get(), storage_type,
            operation, cuda_stream_);
      })
      .and_then([&]() { return output_entity.add<gxf::Tensor>("keypoints3d"); })
      .assign_to(keypoints3d_tensor)
      .and_then([&]() {
        return ToTensor(keypoints3d, keypoints3d_tensor, allocator_.get(), storage_type,
            operation, cuda_stream_);
      })
      .and_then([&]() { return output_entity.add<gxf::Tensor>("position"); })
      .assign_to(position_tensor)
      .and_then([&]() {
        return ToTensor(position, position_tensor, allocator_.get(), storage_type,
            operation, cuda_stream_);
      })
      .and_then([&]() { return output_entity.add<gxf::Tensor>("quaternion_xyzw"); })
      .assign_to(quaternion_tensor)
      .and_then([&]() {
        return ToTensor(quaternion, quaternion_tensor, allocator_.get(), storage_type,
            operation, cuda_stream_);
      })
      .and_then([&]() { return output_entity.add<gxf::Tensor>("score"); })
      .assign_to(score_tensor)
      .and_then([&]() {
        return ToTensor(score, score_tensor, allocator_.get(), storage_type,
            operation, cuda_stream_);
      })
      .and_then([&]() { return output_entity.add<gxf::Tensor>("bbox_size"); })
      .assign_to(bbox_size_tensor)
      .and_then([&]() {
        return ToTensor(bbox_size, bbox_size_tensor, allocator_.get(), storage_type,
            operation, cuda_stream_);
      })
      .and_then([&]() { return output_entity.add<gxf::Timestamp>(kTimestampName); })
      .assign_to(output_timestamp)
      .and_then([&]() { *output_timestamp = *input_timestamp; })
      .and_then([&]() -> gxf::Expected<void> {
        auto cuda_error = cudaStreamSynchronize(cuda_stream_);
        if (cuda_error != cudaSuccess) {
          return gxf::Unexpected{GXF_FAILURE};
        }
        return output_->publish(output_entity);
      });
}

gxf_result_t CenterPosePostProcessor::stop() {
  return GXF_SUCCESS;
}

}  // namespace centerpose
}  // namespace isaac
}  // namespace nvidia
