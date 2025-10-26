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
#include "extensions/centerpose/components/centerpose_detection_to_isaac.hpp"

#include <string>
#include <utility>
#include <vector>

#include "detection3_d_array_message.hpp"
#include "gxf/std/tensor.hpp"

namespace nvidia {
namespace isaac {
namespace centerpose {
namespace {

template <typename T>
struct PointerView {
  PointerView() : data{}, size{} {}
  explicit PointerView(const std::vector<T>& vec) : data{vec.data()}, size{vec.size()} {}

  explicit PointerView(gxf::Handle<gxf::Tensor> tensor)
      : data{tensor->data<T>().value()}, size{tensor->size() / sizeof(T)} {}

  const T* data{};
  size_t size{};
};

template <typename T>
gxf::Expected<std::vector<T>> ToVector(
    gxf::Handle<gxf::Tensor> tensor, const cudaStream_t cuda_stream_) {
  std::vector<T> vec;
  vec.resize(tensor->size() / sizeof(T));
  switch (tensor->storage_type()) {
    case gxf::MemoryStorageType::kHost:
    case gxf::MemoryStorageType::kSystem: {
      std::memcpy(vec.data(), tensor->pointer(), tensor->size());
    } break;
    case gxf::MemoryStorageType::kDevice: {
      // Add cuda stream here aboveand make it an async function call
      const cudaError_t cuda_error = cudaMemcpyAsync(
          vec.data(), tensor->pointer(), tensor->size(), cudaMemcpyDeviceToHost, cuda_stream_);
      if (cuda_error != cudaSuccess) {
        GXF_LOG_ERROR(
            "Failed to transfer memory from device to host: %s", cudaGetErrorString(cuda_error));
        return gxf::Unexpected{GXF_FAILURE};
      }
    } break;
    default:
      GXF_LOG_ERROR("Recieved unexpected MemoryStorageType");
      return gxf::Unexpected{GXF_FAILURE};
  }
  return vec;
}

gxf::Expected<void> CopyIntoDetection3D(
    Detection3DListMessageParts message, PointerView<int64_t> /* class_ids */,
    PointerView<float> positions, PointerView<float> quaternions, PointerView<float> scores,
    PointerView<float> bbox_sizes, const std::string& object_name, const int32_t n_hypothesis) {
  constexpr size_t kNumPointsPosition{3};
  constexpr size_t kNumPointsQuaternion{4};
  constexpr size_t kNumPointsBBoxSize{3};
  for (size_t i = 0; i < message.poses.size(); ++i) {
    const size_t size_idx{i * kNumPointsBBoxSize};
    message.bbox_sizes[i].value()->x() = bbox_sizes.data[size_idx + 0];
    message.bbox_sizes[i].value()->y() = bbox_sizes.data[size_idx + 1];
    message.bbox_sizes[i].value()->z() = bbox_sizes.data[size_idx + 2];

    const size_t position_idx{i * kNumPointsPosition};
    const size_t quaternion_idx{i * kNumPointsQuaternion};

    *message.poses[i].value() = ::nvidia::isaac::Pose3d{
        ::nvidia::isaac::SO3d::FromQuaternion(::nvidia::isaac::Quaterniond{
            quaternions.data[quaternion_idx + 3], quaternions.data[quaternion_idx + 0],
            quaternions.data[quaternion_idx + 1], quaternions.data[quaternion_idx + 2]}),
        ::nvidia::isaac::Vector3d(
            positions.data[position_idx + 0], positions.data[position_idx + 1],
            positions.data[position_idx + 2])};

    for (int32_t j = 0; j < n_hypothesis; ++j) {
      message.hypothesis[i].value()->class_ids.push_back(object_name);
      message.hypothesis[i].value()->scores.push_back(scores.data[i * n_hypothesis + j]);
    }
  }
  return gxf::Success;
}

gxf::Expected<void> ToDetection3DList(
    Detection3DListMessageParts message, gxf::Handle<gxf::Tensor> class_id_tensor,
    gxf::Handle<gxf::Tensor> position_tensor, gxf::Handle<gxf::Tensor> quaternion_tensor,
    gxf::Handle<gxf::Tensor> score_tensor, gxf::Handle<gxf::Tensor> bbox_size_tensor,
    const std::string& object_name, const cudaStream_t cuda_stream) {
  PointerView<int64_t> class_ids_view;
  PointerView<float> positions_view;
  PointerView<float> quaternions_view;
  PointerView<float> scores_view;
  PointerView<float> bbox_sizes_view;

  // Ensure that the vectors don't go out of scope
  std::vector<int64_t> class_ids;
  std::vector<float> positions, quaternions, scores, bbox_sizes;

  switch (class_id_tensor->storage_type()) {
    case gxf::MemoryStorageType::kDevice: {
      auto maybe_vector_result =
          ToVector<int64_t>(class_id_tensor, cuda_stream)
              .assign_to(class_ids)
              .and_then([&]() { return ToVector<float>(position_tensor, cuda_stream); })
              .assign_to(positions)
              .and_then([&]() { return ToVector<float>(quaternion_tensor, cuda_stream); })
              .assign_to(quaternions)
              .and_then([&]() { return ToVector<float>(score_tensor, cuda_stream); })
              .assign_to(scores)
              .and_then([&]() { return ToVector<float>(bbox_size_tensor, cuda_stream); })
              .assign_to(bbox_sizes);
      if (!maybe_vector_result) {
        GXF_LOG_ERROR("Failed to convert tensors into vectors!");
        return gxf::ForwardError(maybe_vector_result);
      }
      auto cuda_error = cudaStreamSynchronize(cuda_stream);
      if (cuda_error != cudaSuccess) {
        GXF_LOG_ERROR("Failed to synchronize stream: %s", cudaGetErrorString(cuda_error));
        return gxf::Unexpected{GXF_FAILURE};
      }
      class_ids_view = PointerView<int64_t>{class_ids};
      positions_view = PointerView<float>{positions};
      quaternions_view = PointerView<float>{quaternions};
      scores_view = PointerView<float>{scores};
      bbox_sizes_view = PointerView<float>{bbox_sizes};
    } break;
    case gxf::MemoryStorageType::kHost:
    case gxf::MemoryStorageType::kSystem: {
      class_ids_view = PointerView<int64_t>{class_id_tensor};
      positions_view = PointerView<float>{position_tensor};
      quaternions_view = PointerView<float>{quaternion_tensor};
      scores_view = PointerView<float>{score_tensor};
      bbox_sizes_view = PointerView<float>{bbox_size_tensor};
    } break;
    default:
      return gxf::Unexpected{GXF_FAILURE};
  }
  return CopyIntoDetection3D(
      message, class_ids_view, positions_view, quaternions_view, scores_view, bbox_sizes_view,
      object_name, score_tensor->shape().dimension(1));
}

}  // namespace

gxf_result_t CenterPoseDetectionToIsaac::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(input_, "input", "Input", "The input");
  result &= registrar->parameter(output_, "output", "Output", "The output");
  result &= registrar->parameter(
      object_name_, "object_name", "Object Name", "The name of the object detected");
  result &= registrar->parameter(
      cuda_stream_pool_, "stream_pool", "Cuda Stream Pool",
      "Instance of gxf::CudaStreamPool to allocate CUDA stream.");
  return gxf::ToResultCode(result);
}

gxf_result_t CenterPoseDetectionToIsaac::start() {
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

gxf_result_t CenterPoseDetectionToIsaac::tick() {
  auto maybe_tensor_entity = input_->receive();
  if (!maybe_tensor_entity) {
    GXF_LOG_ERROR("Failed to receive input message!");
    return gxf::ToResultCode(maybe_tensor_entity);
  }
  gxf::Entity tensor_entity = maybe_tensor_entity.value();
  gxf::Handle<gxf::Tensor> class_id_tensor, position_tensor, quaternion_tensor, score_tensor,
      bbox_size_tensor;
  gxf::Handle<gxf::Timestamp> input_timestamp;

  auto getting_tensor_result =
      tensor_entity.get<gxf::Tensor>("class_id")
          .assign_to(class_id_tensor)
          .and_then([&]() { return tensor_entity.get<gxf::Tensor>("position"); })
          .assign_to(position_tensor)
          .and_then([&]() { return tensor_entity.get<gxf::Tensor>("quaternion_xyzw"); })
          .assign_to(quaternion_tensor)
          .and_then([&]() { return tensor_entity.get<gxf::Tensor>("score"); })
          .assign_to(score_tensor)
          .and_then([&]() { return tensor_entity.get<gxf::Tensor>("bbox_size"); })
          .assign_to(bbox_size_tensor)
          .and_then([&]() { return tensor_entity.get<gxf::Timestamp>("timestamp"); })
          .assign_to(input_timestamp);
  if (!getting_tensor_result) {
    GXF_LOG_ERROR("Failed to get all required tensors");
    return gxf::ToResultCode(getting_tensor_result);
  }
  auto maybe_detection3_d_list =
      CreateDetection3DListMessage(context(), class_id_tensor->shape().dimension(0));
  if (!maybe_detection3_d_list) {
    GXF_LOG_ERROR("Failed to create detection3d list");
    return gxf::ToResultCode(maybe_detection3_d_list);
  }

  if (class_id_tensor->shape().dimension(0) != 0) {
    auto maybe_result = ToDetection3DList(
        maybe_detection3_d_list.value(), class_id_tensor, position_tensor, quaternion_tensor,
        score_tensor, bbox_size_tensor, object_name_.get(), cuda_stream_);
    if (!maybe_result) {
      GXF_LOG_ERROR("Failed to transfer data into Isaac!");
      return gxf::ToResultCode(maybe_result);
    }
  }

  *maybe_detection3_d_list.value().timestamp = *input_timestamp;
  return gxf::ToResultCode(output_->publish(maybe_detection3_d_list.value().entity));
}

gxf_result_t CenterPoseDetectionToIsaac::stop() {
  return GXF_SUCCESS;
}

}  // namespace centerpose
}  // namespace isaac
}  // namespace nvidia
