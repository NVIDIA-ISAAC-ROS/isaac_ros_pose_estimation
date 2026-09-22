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

#include "isaac_ros_foundationpose/foundationpose_impl/pose_decoder.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include <algorithm>

#include "isaac_ros_common/cuda_stream.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace foundationpose
{

namespace
{
constexpr size_t kMatSize = 4;

int findMaxScoreIndex(cudaStream_t stream, const float * scores_device, int N)
{
  std::vector<float> scores_host(N);
  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(
      scores_host.data(), scores_device,
      N * sizeof(float), cudaMemcpyDeviceToHost, stream),
    "memcpy scores D2H");
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream), "sync scores");
  return static_cast<int>(
    std::max_element(scores_host.begin(), scores_host.end()) - scores_host.begin());
}

DecodeResult buildResult(
  const Eigen::Matrix4f & pose_matrix,
  std::shared_ptr<const MeshData> mesh_data,
  const RosMessageStamp & stamp)
{
  Eigen::Matrix4f tf_to_center = Eigen::Matrix4f::Identity();
  tf_to_center.block<3, 1>(0, 3) = -mesh_data->mesh_model_center;
  Eigen::Matrix4f corrected = pose_matrix * tf_to_center;

  Eigen::Matrix4d posed = corrected.cast<double>();
  Eigen::Vector3d translation = posed.col(3).head(3);
  Eigen::Quaterniond rotation(posed.block<3, 3>(0, 0));

  float bbox_x = std::abs(mesh_data->max_vertex[0] - mesh_data->min_vertex[0]);
  float bbox_y = std::abs(mesh_data->max_vertex[1] - mesh_data->min_vertex[1]);
  float bbox_z = std::abs(mesh_data->max_vertex[2] - mesh_data->min_vertex[2]);

  vision_msgs::msg::Detection3D det;
  det.header.stamp.sec = stamp.sec;
  det.header.stamp.nanosec = stamp.nanosec;
  det.header.frame_id = stamp.frame_id;
  det.bbox.center.position.x = translation[0];
  det.bbox.center.position.y = translation[1];
  det.bbox.center.position.z = translation[2];
  det.bbox.center.orientation.w = rotation.w();
  det.bbox.center.orientation.x = rotation.x();
  det.bbox.center.orientation.y = rotation.y();
  det.bbox.center.orientation.z = rotation.z();
  det.bbox.size.x = bbox_x;
  det.bbox.size.y = bbox_y;
  det.bbox.size.z = bbox_z;

  vision_msgs::msg::ObjectHypothesisWithPose hyp;
  hyp.pose.pose = det.bbox.center;
  det.results.push_back(hyp);

  DecodeResult result;
  result.detection3d_array.header.stamp.sec = stamp.sec;
  result.detection3d_array.header.stamp.nanosec = stamp.nanosec;
  result.detection3d_array.header.frame_id = stamp.frame_id;
  result.detection3d_array.detections.push_back(det);
  result.pose_matrix = pose_matrix;
  return result;
}
}  // namespace

PoseDecoder::PoseDecoder(cudaStream_t stream)
: stream_(stream)
{
}

DecodeResult PoseDecoder::decode(
  DevicePoseBatchView poses,
  DeviceScoreBatchView scores,
  std::shared_ptr<const MeshData> mesh_data,
  const RosMessageStamp & stamp)
{
  if (poses.count == 0 || scores.count != poses.count) {
    throw std::runtime_error("[PoseDecoder] decode called with 0 poses");
  }

  int best_idx = findMaxScoreIndex(stream_, scores.data, scores.count);

  Eigen::Matrix4f pose_matrix;
  const size_t mat_bytes = kMatSize * kMatSize * sizeof(float);
  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(
      pose_matrix.data(),
      reinterpret_cast<const char *>(poses.data) + best_idx * mat_bytes,
      mat_bytes, cudaMemcpyDeviceToHost, stream_),
    "memcpy best pose");
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_), "sync best pose");

  return buildResult(pose_matrix, mesh_data, stamp);
}

DecodeResult PoseDecoder::decodeTracking(
  DevicePoseBatchView poses,
  std::shared_ptr<const MeshData> mesh_data,
  const RosMessageStamp & stamp)
{
  if (poses.count != 1) {
    throw std::runtime_error("[PoseDecoder] tracking requires exactly one pose");
  }
  Eigen::Matrix4f pose_matrix;
  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(
      pose_matrix.data(), poses.data,
      kMatSize * kMatSize * sizeof(float), cudaMemcpyDeviceToHost, stream_),
    "memcpy tracking pose");
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_), "sync tracking pose");

  return buildResult(pose_matrix, mesh_data, stamp);
}

}  // namespace foundationpose
}  // namespace isaac_ros
}  // namespace nvidia
