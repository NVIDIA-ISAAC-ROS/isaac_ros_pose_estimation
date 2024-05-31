// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "foundationpose_decoder.hpp"
#include "foundationpose_decoder.cu.hpp"
#include "foundationpose_utils.hpp"

#include "detection3_d_array_message.hpp"
#include "gxf/core/parameter_parser_std.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace isaac_ros {

namespace {

constexpr char kNamePoses[] = "poses";
constexpr size_t kPoseMatrixLength = 4;

}  // namepsace

gxf_result_t FoundationposeDecoder::ExtractBoundingBoxDimensions() {
  // Load the obj mesh file to extrac bbox dimensions
  Assimp::Importer importer;
  const aiScene* scene = importer.ReadFile(
      mesh_file_path_.get(),
      aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);

  if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
    GXF_LOG_ERROR("[FoundationPoseDecoder] Error while loading mesh file ERROR::ASSIMP::");
    GXF_LOG_ERROR(importer.GetErrorString());
    return GXF_FAILURE;
  }

  if (scene->mNumMeshes == 0) {
    GXF_LOG_ERROR("[FoundationPoseDecoder] No mesh was found in the mesh file");
    return GXF_FAILURE;
  }

  const aiMesh* mesh = scene->mMeshes[0];

  auto min_max_vertex = FindMinMaxVertex(mesh);
  mesh_model_center_ = (min_max_vertex.second + min_max_vertex.first) / 2.0;
  Eigen::Vector3f min_vertex = min_max_vertex.first;
  Eigen::Vector3f max_vertex = min_max_vertex.second;

  // Now you have the bounding box defined by the min and max coordinates
  bbox_size_x_ = abs(max_vertex[0] - min_vertex[0]);
  bbox_size_y_ = abs(max_vertex[1] - min_vertex[1]);
  bbox_size_z_ = abs(max_vertex[2] - min_vertex[2]);

  GXF_LOG_INFO(
      "[FoundationposeDecoder] bbox_size_x_: %f \n bbox_size_y_: %f \n bbox_size_z_: %f)",
      bbox_size_x_, bbox_size_y_, bbox_size_z_);
  return GXF_SUCCESS;
}

gxf_result_t FoundationposeDecoder::registerInterface(gxf::Registrar* registrar) noexcept {
  gxf::Expected<void> result;

  result &= registrar->parameter(
      pose_array_receiver_, "pose_array_input", "Pose Array Input",
      "The pose detections array as a tensor");

  result &= registrar->parameter(
      pose_scores_receiver_, "pose_scores_input", "Pose Scores Input",
      "The pose detection scores as a tensor",
      gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);

  result &= registrar->parameter(
      pose_matrix_transmitter_, "pose_matrix_output", "Pose matrix output need for tracking",
      "The ouput pose matrix as a tensor list");

  result &= registrar->parameter(
      detection3_d_list_transmitter_, "output", "Detection 3D list output",
      "The ouput poses as a Detection 3D list");

  result &= registrar->parameter(
      mesh_file_path_, "mesh_file_path", "obj mesh file path", "Path to your object mesh file");

  result &= registrar->parameter(
      mode_, "mode", "Decoder Mode", "Tracking or Pose Estimation");

  result &= registrar->parameter(
      allocator_, "allocator", "Allocator", "Output Allocator");

  result &= registrar->parameter(
      cuda_stream_pool_, "cuda_stream_pool", "Cuda Stream Pool",
      "Instance of gxf::CudaStreamPool to allocate CUDA stream.");
  return gxf::ToResultCode(result);
}

gxf_result_t FoundationposeDecoder::start() noexcept {
  // Get cuda stream from stream pool
  auto maybe_stream = cuda_stream_pool_.get()->allocateStream();
  if (!maybe_stream) { return gxf::ToResultCode(maybe_stream); }

  cuda_stream_handle_ = std::move(maybe_stream.value());
  if (!cuda_stream_handle_->stream()) {
    GXF_LOG_ERROR("[FoundationposeDecoder] Allocated stream is not initialized!");
    return GXF_FAILURE;
  }
  if (!cuda_stream_handle_.is_null()) {
    cuda_stream_ = cuda_stream_handle_->stream().value();
  }

  return ExtractBoundingBoxDimensions();
}

gxf_result_t FoundationposeDecoder::tick() noexcept {
  GXF_LOG_DEBUG("[FoundationposeDecoder] Tick");

  // Receive pose array data and score tensors
  const auto maybe_pose_array_message = pose_array_receiver_->receive();
  if (!maybe_pose_array_message) {
    GXF_LOG_ERROR("[FoundationposeDecoder] Failed to get pose array message from receiver.");
    return gxf::ToResultCode(maybe_pose_array_message);
  }
  auto maybe_pose_array_tensor = maybe_pose_array_message.value().get<gxf::Tensor>();
  if (!maybe_pose_array_tensor) {
    GXF_LOG_ERROR("[FoundationposeDecoder] Failed to get pose array tensor from the message.");
    return gxf::ToResultCode(maybe_pose_array_tensor);
  }
  auto pose_array_tensor = maybe_pose_array_tensor.value();
  
  auto n_detections = pose_array_tensor->shape().dimension(0);
  if (n_detections == 0) {
    GXF_LOG_ERROR("[FoundationposeDecoder] Zero sample received from sampling node");
    return GXF_FAILURE;
  }
  GXF_LOG_DEBUG("[FoundationposeDecoder] Number of poses recevied: %d", n_detections);

  int best_score_index = 0;
  if (mode_.get()!= "tracking") {
    auto pose_scores_receiver = pose_scores_receiver_.try_get();
    if (!pose_scores_receiver) {
      GXF_LOG_ERROR("[FoundationposeDecoder] Failed to get pose score receiver.");
      return gxf::ToResultCode(pose_scores_receiver);
    }
    const auto maybe_pose_scores_message = pose_scores_receiver.value()->receive();
    if (!maybe_pose_scores_message) {
      GXF_LOG_ERROR("[FoundationposeDecoder] Failed to get pose scores message from receiver.");
      return gxf::ToResultCode(maybe_pose_scores_message);
    }
    auto maybe_pose_scores_tensor = maybe_pose_scores_message.value().get<gxf::Tensor>();
    if (!maybe_pose_scores_tensor) {
      GXF_LOG_ERROR("[FoundationposeDecoder] Failed to get pose scores tensor from the message.");
      return gxf::ToResultCode(maybe_pose_scores_tensor);
    }
    auto pose_scores_tensor = maybe_pose_scores_tensor.value();

    // Ensure number of poses in pose array and pose scores are equal
    if (pose_array_tensor->shape().dimension(0) != pose_scores_tensor->shape().dimension(1)) {
      GXF_LOG_ERROR(
        "[FoundationposeDecoder] Number of poses in pose array(%d) and pose scores(%d) are not "
        "equal",
        pose_array_tensor->shape().dimension(0), pose_scores_tensor->shape().dimension(1));
      return GXF_FAILURE;
    }

    // Find the index of the highest score if there are multiple detections
    best_score_index =
      getMaxScoreIndex(cuda_stream_, reinterpret_cast<float*>(pose_scores_tensor->pointer()), n_detections);
    CHECK_CUDA_ERRORS(cudaGetLastError());
    GXF_LOG_DEBUG("[FoundationposeDecoder] Selected index from the score model: %d", best_score_index);
  }

  // Extract the pose matrix with highest score
  Eigen::Matrix4f pose_matrix;
  int bytes_per_element = 4 * 4 * sizeof(float);
  CHECK_CUDA_ERRORS(cudaMemcpyAsync(
      pose_matrix.data(), pose_array_tensor->pointer() + best_score_index * bytes_per_element,
      bytes_per_element, cudaMemcpyDeviceToHost, cuda_stream_));
  CHECK_CUDA_ERRORS(cudaStreamSynchronize(cuda_stream_));
  
  // Add the distance from edge to the center because
  Eigen::Matrix4f tf_to_center = Eigen::Matrix4f::Identity();
  tf_to_center.block<3, 1>(0, 3) = mesh_model_center_;
  pose_matrix = pose_matrix * tf_to_center;
  

  // Publish the pose matrix as the input of next frame for tracking mode
  auto maybe_pose_matrix_message = gxf::Entity::New(context());
  if (!maybe_pose_matrix_message) {
    GXF_LOG_ERROR("[FoundationposeDecoder] Failed to allocate PoseArray Message");
    return gxf::ToResultCode(maybe_pose_matrix_message);
  }
  auto pose_matrix_message = maybe_pose_matrix_message.value();

  // Pose Array is column-major
  auto maybe_pose_arrays = pose_matrix_message.add<gxf::Tensor>(kNamePoses);
  if (!maybe_pose_arrays) {
    GXF_LOG_ERROR("[FoundationposeDecoder] Failed to allocate output Tensor ");
    return gxf::ToResultCode(maybe_pose_arrays);
  }
  auto pose_arrays = maybe_pose_arrays.value();

  auto maybe_added_timestamp_pose_matrix =
      AddInputTimestampToOutput(pose_matrix_message, maybe_pose_array_message.value());
  if (!maybe_added_timestamp_pose_matrix) {
    GXF_LOG_ERROR("[FoundationposeDecoder] Failed to add timestamp");
    return gxf::ToResultCode(maybe_added_timestamp_pose_matrix);
  }

  std::array<int32_t, nvidia::gxf::Shape::kMaxRank> pose_arrays_shape{
      1, kPoseMatrixLength, kPoseMatrixLength};
  // Initializing GXF tensor
  auto result = pose_arrays->reshape<float>(
      nvidia::gxf::Shape{pose_arrays_shape, 3}, nvidia::gxf::MemoryStorageType::kDevice,
      allocator_);
  if (!result) {
    GXF_LOG_ERROR("[FoundationposeDecoder] Failed to reshape pose array tensor");
    return gxf::ToResultCode(result);
  }
  CHECK_CUDA_ERRORS(cudaMemcpyAsync(
      pose_arrays->pointer(), pose_matrix.data(), pose_arrays->size(), cudaMemcpyHostToDevice, cuda_stream_));
  CHECK_CUDA_ERRORS(cudaStreamSynchronize(cuda_stream_));
  pose_matrix_transmitter_->publish(std::move(pose_matrix_message));


  // Convert pose matrix into translation and rotation
  Eigen::Matrix4d pose_matrixd = pose_matrix.cast<double>();

  Eigen::Vector3d translation = pose_matrixd.col(3).head(3);
  Eigen::Matrix3d rotation_matrix = pose_matrixd.block<3, 3>(0, 0);
  Eigen::Quaterniond rotation(rotation_matrix);

  // Prepare outputs and publish
  auto maybe_detection3_d_list = nvidia::isaac::CreateDetection3DListMessage(context(), 1);
  if (!maybe_detection3_d_list) {
    GXF_LOG_ERROR("[FoundationposeDecoder] Failed to create detection3d list");
    return gxf::ToResultCode(maybe_detection3_d_list);
  }

  auto maybe_added_timestamp =
      AddInputTimestampToOutput(maybe_detection3_d_list->entity, maybe_pose_array_message.value());
  if (!maybe_added_timestamp) {
    GXF_LOG_ERROR("[FoundationposeDecoder] Failed to add timestamp");
    return gxf::ToResultCode(maybe_added_timestamp);
  }
  auto detection3_d_list = maybe_detection3_d_list.value();

  nvidia::isaac::Pose3d pose3d{
      nvidia::isaac::SO3d::FromQuaternion({rotation.w(), rotation.x(), rotation.y(), rotation.z()}),
      nvidia::isaac::Vector3d(translation[0], translation[1], translation[2])};

  **detection3_d_list.poses[0] = pose3d;
  **detection3_d_list.bbox_sizes[0] =
      nvidia::isaac::Vector3f(bbox_size_x_, bbox_size_y_, bbox_size_z_);

  return gxf::ToResultCode(
      detection3_d_list_transmitter_->publish(std::move(detection3_d_list.entity)));
}

}  // namespace isaac_ros
}  // namespace nvidia
