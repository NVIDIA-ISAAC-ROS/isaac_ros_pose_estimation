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

#include "foundationpose_transformation.hpp"

#include <cmath>
#include <iostream>
#include <string>

#include <cuda_runtime.h>

#include "foundationpose_utils.hpp"
#include "gxf/multimedia/camera.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace isaac_ros {

namespace {

constexpr char kNamePoses[] = "poses";
constexpr char kNameTranslations[] = "output_tensor1";
constexpr char kNameRotations[] = "output_tensor2";
constexpr size_t kTransformationMatrixSize = 4;
constexpr int kNumBatches = 6;

}  // namespace

gxf_result_t FoundationposeTransformation::registerInterface(gxf::Registrar* registrar) noexcept {
  gxf::Expected<void> result;

  result &= registrar->parameter(
      poses_receiver_, "poses_input", "Pose Array Input", "The pose detections array as a tensor");

  result &= registrar->parameter(
      refined_poses_receiver_, "refined_poses_input", "Refined Pose detla from refinere network",
      "The poses detection array as two tensors");

  result &= registrar->parameter(
      sliced_pose_array_transmitter_, "sliced_output", "Pose_Array output", "Sliced ouput poses as a pose array");
  
  result &= registrar->parameter(
      batched_pose_array_transmitter_, "batched_output", "Pose_Array output", "Batched ouput poses as a pose array");

  result &= registrar->parameter(
      rot_normalizer_, "rot_normalizer", "Rotation Normalizer", "Rotation Normalizer");

  result &= registrar->parameter(
      mode_, "mode", "Transformation Mode", "Tracking or Pose Estimation");

  result &= registrar->parameter(
      refine_iterations_, "refine_iterations", "refine iterations", 
      "Number of iterations on the refine network (render->refine->transoformation)", 1);

  result &= registrar->parameter(
      allocator_, "allocator", "Allocator", "Output Allocator");

  result &= registrar->parameter(
      cuda_stream_pool_, "cuda_stream_pool", "Cuda Stream Pool",
      "Instance of gxf::CudaStreamPool to allocate CUDA stream.");

  result &= registrar->parameter(
      mesh_storage_, "mesh_storage", "Mesh Storage",
      "Component to reuse mesh");

  // Only used for iterative refinement
  result &= registrar->parameter(
      iterative_sliced_pose_array_transmitter_, "iterative_sliced_output", "Iterative sliced output",
      "Sliced ouput poses as a pose array sent back to refine pipeline",
      gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);

  result &= registrar->parameter(
      iterative_poses_receiver_, "iterative_poses_input", "Iterative sliced input",
      "Sliced ouput poses as a pose array sent back to itself",
      gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);

  return gxf::ToResultCode(result);
}

gxf_result_t FoundationposeTransformation::start() noexcept {
  // Validate input parameters
  if (refine_iterations_ < 1) {
    GXF_LOG_ERROR("Refine iterations should be at least 1");
    return GXF_FAILURE;
  }

  // Get cuda stream from stream pool
  auto maybe_stream = cuda_stream_pool_.get()->allocateStream();
  if (!maybe_stream) { return gxf::ToResultCode(maybe_stream); }

  cuda_stream_handle_ = std::move(maybe_stream.value());
  if (!cuda_stream_handle_->stream()) {
    GXF_LOG_ERROR("Error: allocated stream is not initialized!");
    return GXF_FAILURE;
  }
  if (!cuda_stream_handle_.is_null()) {
    cuda_stream_ = cuda_stream_handle_->stream().value();
  }
  return GXF_SUCCESS;
}

gxf_result_t FoundationposeTransformation::tick() noexcept {
  GXF_LOG_DEBUG("[FoundationposeTransformation] Tick FoundationposeTransformation");

  const auto maybe_refined_poses_message = refined_poses_receiver_->receive();
  if (!maybe_refined_poses_message) {
    GXF_LOG_ERROR("[FoundationposeTransformation] Refined pose message is not valid");
    return gxf::ToResultCode(maybe_refined_poses_message);
  }

  gxf::Entity pose_message;
  if (iteration_count_ == 0) {
    const auto maybe_pose_message = poses_receiver_->receive();
    if (!maybe_pose_message) {
      GXF_LOG_ERROR("[FoundationposeTransformation] Pose message is not valid");
      return gxf::ToResultCode(maybe_pose_message);
    }
    pose_message = maybe_pose_message.value();
  } else {
    auto iterative_pose_receiver = iterative_poses_receiver_.try_get();
    if (!iterative_pose_receiver) {
      GXF_LOG_ERROR("[FoundationposeTransformation] Failed to get iterative pose receiver.");
      return gxf::ToResultCode(iterative_pose_receiver);
    }
    auto maybe_pose_message = iterative_pose_receiver.value()->receive();
    if (!maybe_pose_message) {
      GXF_LOG_ERROR("[FoundationposeTransformation] Failed to receive iterative pose message");
      return maybe_pose_message.error();
    }
    pose_message = maybe_pose_message.value();
  }

  auto maybe_rotation_tensor = maybe_refined_poses_message.value().get<gxf::Tensor>(kNameRotations);
  if (!maybe_rotation_tensor) {
    GXF_LOG_ERROR("[FoundationposeTransformation] Failed to get rotation tensor.");
    return gxf::ToResultCode(maybe_rotation_tensor);
  }
  auto rotation_tensor = maybe_rotation_tensor.value();

  auto maybe_translation_tensor =
      maybe_refined_poses_message.value().get<gxf::Tensor>(kNameTranslations);
  if (!maybe_translation_tensor) {
    GXF_LOG_ERROR("[FoundationposeTransformation] Failed to get translation tensor.");
    return gxf::ToResultCode(maybe_translation_tensor);
  }

  auto maybe_pose_tensor = pose_message.get<gxf::Tensor>(kNamePoses);
  if (!maybe_pose_tensor) {
    GXF_LOG_ERROR("[FoundationposeTransformation] Failed to get poses from message");
    return maybe_pose_tensor.error();
  }

  // Load and process mesh data
  auto mesh_data_ptr = mesh_storage_.get()->GetMeshData();
  if (!mesh_data_ptr) {
    GXF_LOG_ERROR("[FoundationposeTransformation] Failed to load mesh data");
    return GXF_FAILURE;
  }

  auto rotations_handle = maybe_rotation_tensor.value();
  auto translations_handle = maybe_translation_tensor.value();
  auto poses_handle = maybe_pose_tensor.value();

  const uint32_t pose_nums = poses_handle->shape().dimension(0);
  const uint32_t pose_rows = poses_handle->shape().dimension(1);
  const uint32_t pose_cols = poses_handle->shape().dimension(2);

  const uint32_t rotation_nums = rotations_handle->shape().dimension(0);
  const uint32_t rotation_shape = rotations_handle->shape().dimension(1);

  const uint32_t translations_nums = translations_handle->shape().dimension(0);
  const uint32_t translations_shape = translations_handle->shape().dimension(1);

  // Check the size are equal
  if (pose_nums == 0) {
    GXF_LOG_ERROR("[FoundationposeTransformation] Zero poses received by transformation node");
    return GXF_FAILURE;
  }

  // Check the size are equal
  if (pose_rows != kTransformationMatrixSize || pose_cols != kTransformationMatrixSize) {
    GXF_LOG_ERROR("[FoundationposeTransformation] The transformation matrix is not 4x4!");
    return GXF_FAILURE;
  }

  // Check translation and rotation num are equal
  if (rotation_nums != translations_nums) {
    GXF_LOG_ERROR("[FoundationposeTransformation] The received poses are not equal! rotation_nums: %d, translations_nums: %d", rotation_nums, translations_nums);
    return GXF_FAILURE;
  }

  // Pose num and translation num could be different, as tensorRT allocate output with the max batch size
  // Pose num should always be less or equal to translation num
  if (pose_nums > translations_nums) {
    GXF_LOG_ERROR("[FoundationposeTransformation] The pose num is greater than the translation num! pose_nums: %d, translations_nums: %d", pose_nums, translations_nums);
    return GXF_FAILURE;
  }

  // Create a vector to hold N Eigen 4x4 matrices
  std::vector<float> B_in_cam(pose_nums * pose_rows * pose_cols);
  std::vector<float> trans_delta(pose_nums * translations_shape);
  std::vector<float> rot_delta(pose_nums * rotation_shape);
  std::vector<Eigen::Vector3f> trans_delta_vec(pose_nums);
  std::vector<Eigen::Matrix3f> rot_mat_delta(pose_nums);

  CHECK_CUDA_ERRORS(cudaMemcpyAsync(
      B_in_cam.data(), poses_handle->pointer(), B_in_cam.size() * sizeof(float),
      cudaMemcpyDeviceToHost, cuda_stream_));
  CHECK_CUDA_ERRORS(cudaMemcpyAsync(
      trans_delta.data(), translations_handle->pointer(), trans_delta.size() * sizeof(float),
      cudaMemcpyDeviceToHost, cuda_stream_));
  CHECK_CUDA_ERRORS(cudaMemcpyAsync(
      rot_delta.data(), rotations_handle->pointer(), rot_delta.size() * sizeof(float),
      cudaMemcpyDeviceToHost, cuda_stream_));
  CHECK_CUDA_ERRORS(cudaStreamSynchronize(cuda_stream_));

  // Reconstruct into vector
  for (size_t index = 0; index < pose_nums; index++) {
    Eigen::Map<Eigen::Vector3f> cur_trans_delta(trans_delta.data() + index * 3, 3);
    Eigen::Map<Eigen::Vector3f> cur_rot_delta(rot_delta.data() + index * 3, 3);

    trans_delta_vec[index] = cur_trans_delta.array() * (mesh_data_ptr->mesh_diameter / 2);
    // Transform rotation vector into matrix through Rodrigues transform
    auto normalized_vect = (cur_rot_delta.array().tanh() * rot_normalizer_).matrix();
    Eigen::AngleAxisf rot_delta_angle_axis(normalized_vect.norm(), normalized_vect.normalized());
    rot_mat_delta[index] = rot_delta_angle_axis.toRotationMatrix().transpose();
  }

  // Pose array is colum-major
  std::vector<float> refined_pose;
  refined_pose.reserve(pose_nums * kTransformationMatrixSize * kTransformationMatrixSize);
  for (size_t index = 0; index < pose_nums; index++) {
    // Contruct matrix from vector
    Eigen::Map<Eigen::MatrixXf> cur_pose(
        B_in_cam.data() + index * kTransformationMatrixSize * kTransformationMatrixSize,
        kTransformationMatrixSize, kTransformationMatrixSize);
    // Add the last column of B_in_cam[i] with trans_delta[i]
    cur_pose.col(3).head(3) += trans_delta_vec[index];

    // Extract the top-left 3x3 part of B_in_cam[i]
    Eigen::Matrix3f top_left_3x3 = cur_pose.block<3, 3>(0, 0);
    // Multiply the 3x3 part with rot_mat_delta[i]
    Eigen::Matrix3f result_3x3 = rot_mat_delta[index] * top_left_3x3;
    // Place the result back into the top-left 3x3 part of B_in_cam[i]
    cur_pose.block<3, 3>(0, 0) = result_3x3;

    // flatten matrix back to vector and insert
    std::vector<float> pose_data(cur_pose.data(), cur_pose.data() + cur_pose.size());
    refined_pose.insert(refined_pose.end(), pose_data.begin(), pose_data.end());
  }
  if (iteration_count_ == refine_iterations_ - 1) {
    // Prepare batched poses and sent to the final decoder
    batched_refined_pose_.insert(batched_refined_pose_.end(), refined_pose.begin(), refined_pose.end());
  }

  // Allocate output message
  auto maybe_output_message = gxf::Entity::New(context());
  if (!maybe_output_message) {
    GXF_LOG_ERROR("[FoundationposeTransformation] Failed to allocate PoseArray Message");
    return gxf::ToResultCode(maybe_output_message);
  }
  auto output_message = maybe_output_message.value();

  auto maybe_added_timestamp =
      AddInputTimestampToOutput(output_message, pose_message);
  if (!maybe_added_timestamp) {
    GXF_LOG_ERROR("[FoundationposeTransformation] Failed to add timestamp");
    return gxf::ToResultCode(maybe_added_timestamp);
  }

  auto maybe_pose_arrays = output_message.add<gxf::Tensor>(kNamePoses);
  if (!maybe_pose_arrays) {
    GXF_LOG_ERROR("[FoundationposeTransformation] Failed to allocate output poses");
    return gxf::ToResultCode(maybe_pose_arrays);
  }
  auto pose_arrays = maybe_pose_arrays.value();

  std::array<int32_t, nvidia::gxf::Shape::kMaxRank> pose_arrays_shape{
      static_cast<int>(pose_nums), kTransformationMatrixSize, kTransformationMatrixSize};
  // Initializing GXF tensor
  auto result = pose_arrays->reshape<float>(
      nvidia::gxf::Shape{pose_arrays_shape, 3}, nvidia::gxf::MemoryStorageType::kDevice, allocator_);
  if (!result) {
    GXF_LOG_ERROR("[FoundationposeTransformation] Failed to reshape pose array tensor");
    return gxf::ToResultCode(result);
  }
  CHECK_CUDA_ERRORS(cudaMemcpyAsync(
      pose_arrays->pointer(), refined_pose.data(), pose_arrays->size(),
      cudaMemcpyHostToDevice, cuda_stream_));
  CHECK_CUDA_ERRORS(cudaStreamSynchronize(cuda_stream_));
  
  if (mode_.get() == "tracking") {
    GXF_LOG_DEBUG ("[FoundationposeTransformation] Tracking mode, directly publish output.");
    return gxf::ToResultCode(sliced_pose_array_transmitter_->publish(std::move(output_message)));
  }

  // During iteration: send the sliced message back to refine render
  // On the last iteration but not all messages are accumulated: send the slice message to score render
  // On the last iteration and all messages are accumulated: send the batched message to decoder
  if (iteration_count_ < refine_iterations_ - 1) {
    GXF_LOG_DEBUG ("[FoundationposeTransformation] Publish iterative sliced poses to render.");
    auto iterative_sliced_pose_array_transmitter = iterative_sliced_pose_array_transmitter_.try_get();
    if (!iterative_sliced_pose_array_transmitter) {
      GXF_LOG_ERROR("[FoundationposeTransformation] Failed to get iterative pose array transmitter.");
      return gxf::ToResultCode(iterative_sliced_pose_array_transmitter);
    }
    iterative_sliced_pose_array_transmitter.value()->publish(std::move(output_message));
  } else if (received_batches_ < kNumBatches - 1) {
    // Expected more batches, only publish current sliced message to score render.
    GXF_LOG_DEBUG ("[FoundationposeTransformation] Publish sliced poses to render.");
    sliced_pose_array_transmitter_->publish(std::move(output_message));
  } else {
    // Accumulated all batcheds, publish batched message to decoder and current sliced message to score render.
    GXF_LOG_DEBUG ("[FoundationposeTransformation] Publish batched poses to decoder.");
    auto maybe_batched_output_message = gxf::Entity::New(context());
    if (!maybe_batched_output_message) {
      GXF_LOG_ERROR("[FoundationposeTransformation] Failed to allocate PoseArray Message");
      return gxf::ToResultCode(maybe_batched_output_message);
    }
    auto batched_output_message = maybe_batched_output_message.value();

    auto maybe_added_timestamp =
        AddInputTimestampToOutput(batched_output_message, pose_message);
    if (!maybe_added_timestamp) {
      GXF_LOG_ERROR("[FoundationposeTransformation] Failed to add timestamp");
      return gxf::ToResultCode(maybe_added_timestamp);
    }

    auto maybe_batched_pose_arrays = batched_output_message.add<gxf::Tensor>(kNamePoses);
    if (!maybe_batched_pose_arrays) {
      GXF_LOG_ERROR("[FoundationposeTransformation] Failed to allocate output poses");
      return gxf::ToResultCode(maybe_batched_pose_arrays);
    }
    auto batched_pose_arrays = maybe_batched_pose_arrays.value();

    std::array<int32_t, nvidia::gxf::Shape::kMaxRank> pose_arrays_shape{
        static_cast<int>(pose_nums * kNumBatches), kTransformationMatrixSize, kTransformationMatrixSize};

    // Initializing GXF tensor
    auto result = batched_pose_arrays->reshape<float>(
        nvidia::gxf::Shape{pose_arrays_shape, 3}, nvidia::gxf::MemoryStorageType::kDevice, allocator_);
    if (!result) {
      GXF_LOG_ERROR("[FoundationposeTransformation] Failed to reshape pose array tensor");
      return gxf::ToResultCode(result);
    }
    CHECK_CUDA_ERRORS(cudaMemcpyAsync(
      batched_pose_arrays->pointer(), batched_refined_pose_.data(), batched_pose_arrays->size(),
      cudaMemcpyHostToDevice, cuda_stream_));
    batched_refined_pose_.clear();

    sliced_pose_array_transmitter_->publish(std::move(output_message));
    batched_pose_array_transmitter_->publish(std::move(batched_output_message));
  }

  // Update iteration and batch counters
  received_batches_ += 1;
  if (received_batches_ == kNumBatches) {
    received_batches_ = 0;
    iteration_count_ += 1;
    GXF_LOG_DEBUG ("[FoundationposeTransformation] %d refine iterations are finished", iteration_count_);
  }
  if (iteration_count_ == refine_iterations_) {
    iteration_count_ = 0;
  }
  return GXF_SUCCESS;
}

}  // namespace isaac_ros
}  // namespace nvidia
