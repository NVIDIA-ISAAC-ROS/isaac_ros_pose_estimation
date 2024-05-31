// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dope_decoder.hpp"

#include <Eigen/Dense>

#include <cuda_runtime.h>

#include "opencv2/calib3d.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgproc.hpp"

#include <string>

#include "gxf/multimedia/camera.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/core/parameter_parser_std.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace isaac_ros {
namespace dope {

namespace {
// The dimensions of the input belief map tensor. The two major dimensions are fixed and
// determined by the output size of the DOPE DNN.
constexpr size_t kInputMapsChannels = 25;
constexpr size_t kNumTensors = 1;
// The dimensions of the output pose array tensor:
// position (xyz) and orientation (quaternion, xyzw)
constexpr int kExpectedPoseAsTensorSize = (3 + 4);
// The number of vertex (belief map) channels in the DNN output tensor for the 8
// corners and 1 centroid. The other channels are affinity maps (vector fields)
// for the 8 corners.
constexpr size_t kNumCorners = 8;
constexpr size_t kNumVertexChannel = kNumCorners + 1;
// The standard deviation of the Gaussian blur
constexpr float kGaussianSigma = 3.0;
// Minimum acceptable sum of averaging weights
constexpr float kMinimumWeightSum = 1e-6;
constexpr float kMapPeakThreshhold = 0.01;
constexpr float kPeakWeightThreshhold = 0.1;
constexpr float kAffinityMapAngleThreshhold = 0.5;
// Offset added to belief map pixel coordinate, constant for the fixed input
// image size
// https://github.com/NVlabs/Deep_Object_Pose/blob/master/src/dope/inference/detector.py
// line 343
constexpr float kOffsetDueToUpsampling = 0.4395f;
// The original image is kImageToMapScale larger in each dimension than the
// output tensor from the DNN.
constexpr float kImageToMapScale = 8.0f;
// From the requirements listed for pnp::ComputeCameraPoseEpnp.
constexpr size_t kRequiredPointsForPnP = 6;
// Placeholder for unidentify peak ids in DopeObject
constexpr int kInvalidId = -1;
// Placeholder for unknown best distance from centroid to peak in DopeObject
constexpr float kInvalidDist = std::numeric_limits<float>::max();

// The list of keypoint indices (0-8, 8 is centroid) and their corresponding 2d
// pixel coordinates as columns in a matrix.
using DopeObjectKeypoints = std::pair<std::vector<int>, Eigen::Matrix2Xf>;

// An internal class used to store information about detected objects. Used only
// within the 'FindObjects' function.
struct DopeObject {
  explicit DopeObject(int id) : center(id) {
    for (size_t ii = 0; ii < kNumCorners; ++ii) {
      corners[ii] = kInvalidId;
      best_distances[ii] = kInvalidDist;
    }
  }

  int center;
  std::array<int, kNumCorners> corners;
  std::array<float, kNumCorners> best_distances;
};

struct Pose3d {
public:
  Eigen::Vector3d translation;
  Eigen::Quaterniond rotation;

  Pose3d inverse() {
    Pose3d retval;
    retval.translation = -this->translation;
    retval.rotation = this->rotation.inverse();
    return retval;
  }
};

// Returns pixel mask for local maximums in single - channel image src
void IsolateMaxima(const cv::Mat &src, cv::Mat &mask) {
  // Find pixels that are equal to the local neighborhood maxima
  cv::dilate(src, mask, cv::Mat());
  cv::compare(src, mask, mask, cv::CMP_GE);

  // Filter out pixels that are not equal to the local maximum ('plateaus')
  cv::Mat non_plateau_mask;
  cv::erode(src, non_plateau_mask, cv::Mat());
  cv::compare(src, non_plateau_mask, non_plateau_mask, cv::CMP_GT);
  cv::bitwise_and(mask, non_plateau_mask, mask);
}

// Returns pixel coordinate (row, col) of maxima in single-channel image
std::vector<Eigen::Vector2i> FindPeaks(const cv::Mat &image,
                                       const float threshold) {
  // Extract centers of local maxima
  cv::Mat mask;
  std::vector<cv::Point> maxima;
  IsolateMaxima(image, mask);
  cv::findNonZero(mask, maxima);

  // Find maxima
  std::vector<Eigen::Vector2i> peaks;
  for (const auto &m : maxima) {
    if (image.at<float>(m.y, m.x) > threshold) {
      peaks.push_back(Eigen::Vector2i(m.x, m.y));
    }
  }

  return peaks;
}

// Returns 3x9 matrix of the 3d coordinates of cuboid corners and center
Eigen::Matrix<double, 3, kNumVertexChannel>
CuboidVertices(const std::array<double, 3> &ext) {
  // X axis points to the right
  const double right = ext.at(0) * 0.5;
  const double left = -ext.at(0) * 0.5;
  // Y axis points downward
  const double bottom = ext.at(1) * 0.5;
  const double top = -ext.at(1) * 0.5;
  // Z axis points forward (away from camera)
  const double front = ext.at(2) * 0.5;
  const double rear = -ext.at(2) * 0.5;

  Eigen::Matrix<double, 3, kNumVertexChannel> points;
  points << right, left, left, right, right, left, left, right, 0.0, top, top,
      bottom, bottom, top, top, bottom, bottom, 0.0, front, front, front, front,
      rear, rear, rear, rear, 0.0;

  return points;
}

std::vector<DopeObjectKeypoints>
FindObjects(const std::array<cv::Mat, kInputMapsChannels> &maps) {
  using Vector2f = Eigen::Vector2f;
  using Vector2i = Eigen::Vector2i;

  // 'all_peaks' contains: x,y: 2d location of peak; z: belief map value
  std::vector<Vector2f> all_peaks;
  std::array<std::vector<int>, kNumVertexChannel> channel_peaks;
  cv::Mat image{};
  for (size_t chan = 0; chan < kNumVertexChannel; ++chan) {
    // Isolate and copy a single channel
    image = maps[chan].clone();

    // Smooth the image
    cv::Mat blurred;
    cv::GaussianBlur(image, blurred, cv::Size(0, 0), kGaussianSigma,
                     kGaussianSigma, cv::BORDER_REFLECT);

    // Find the maxima of the tensor values in this channel
    std::vector<Vector2i> peaks = FindPeaks(blurred, kMapPeakThreshhold);
    for (size_t pp = 0; pp < peaks.size(); ++pp) {
      const auto peak = peaks[pp];

      // Compute the weighted average for localizing the peak, using a 5x5
      // window
      Vector2f peak_sum(0, 0);
      float weight_sum = 0.0f;
      for (int ii = -2; ii <= 2; ++ii) {
        for (int jj = -2; jj <= 2; ++jj) {
          const int row = peak[0] + ii;
          const int col = peak[1] + jj;

          if (col < 0 || col >= image.size[1] || row < 0 ||
              row >= image.size[0]) {
            continue;
          }

          const float weight = image.at<float>(row, col);
          weight_sum += weight;
          peak_sum[0] += row * weight;
          peak_sum[1] += col * weight;
        }
      }

      if (image.at<float>(peak[1], peak[0]) >= kMapPeakThreshhold) {
        channel_peaks[chan].push_back(static_cast<int>(all_peaks.size()));
        if (std::fabs(weight_sum) < kMinimumWeightSum) {
          all_peaks.push_back({peak[0] + kOffsetDueToUpsampling,
                               peak[1] + kOffsetDueToUpsampling});
        } else {
          all_peaks.push_back(
              {peak_sum[0] / weight_sum + kOffsetDueToUpsampling,
               peak_sum[1] / weight_sum + kOffsetDueToUpsampling});
        }
      }
    }
  }

  // Create a list of potential objects using the detected centroid peaks (the
  // 9th channel results above)
  std::vector<DopeObject> objects;
  for (auto peak : channel_peaks[kNumVertexChannel - 1]) {
    objects.push_back(DopeObject{peak});
  }

  // Use 16 affinity field tensors (2 for each corner to centroid) to identify
  // corner-centroid associated for each corner peak
  for (size_t chan = 0; chan < kNumVertexChannel - 1; ++chan) {
    const std::vector<int> &peaks = channel_peaks[chan];
    for (size_t pp = 0; pp < peaks.size(); ++pp) {
      int best_idx = kInvalidId;
      float best_distance = kInvalidDist;
      float best_angle = kInvalidDist;

      for (size_t jj = 0; jj < objects.size(); ++jj) {
        const Vector2f &center = all_peaks[objects[jj].center];
        const Vector2f &point = all_peaks[peaks[pp]];
        const Vector2i point_int(static_cast<int>(point[0]),
                                 static_cast<int>(point[1]));

        Vector2f v_aff(maps[kNumVertexChannel + chan * 2].at<float>(
                           point_int[1], point_int[0]),
                       maps[kNumVertexChannel + chan * 2 + 1].at<float>(
                           point_int[1], point_int[0]));
        v_aff.normalize();

        const Vector2f v_center = (center - point).normalized();

        const float angle = (v_center - v_aff).norm();
        const float dist = (point - center).norm();

        if (angle < kAffinityMapAngleThreshhold && dist < best_distance) {
          best_idx = jj;
          best_distance = dist;
          best_angle = angle;
        }
      }
      // Cannot find a centroid to associate this corner peak with
      if (best_idx == kInvalidId) {
        continue;
      }

      if (objects[best_idx].corners[chan] == kInvalidId ||
          (best_angle < kAffinityMapAngleThreshhold &&
           best_distance < objects[best_idx].best_distances[chan])) {
        objects[best_idx].corners[chan] = peaks[pp];
        objects[best_idx].best_distances[chan] = best_distance;
      }
    }
  }

  std::vector<DopeObjectKeypoints> output;
  for (const DopeObject &object : objects) {
    // Get list of indices of valid corners in object
    std::vector<int> valid_indices;
    for (size_t ii = 0; ii < object.corners.size(); ii++) {
      if (object.corners[ii] != kInvalidId) {
        valid_indices.push_back(ii);
      }
    }

    // Centroid is always valid
    valid_indices.push_back(kNumVertexChannel - 1);
    const size_t num_valid = valid_indices.size();

    // If we don't have enough valid points for PnP, skip it
    if (num_valid < kRequiredPointsForPnP) {
      continue;
    }

    // Collect 2d image pixel coordinates of valid peaks
    Eigen::Matrix2Xf image_coordinates(2, num_valid);
    for (size_t ii = 0; ii < num_valid - 1; ++ii) {
      image_coordinates.col(ii) =
          all_peaks[object.corners[valid_indices[ii]]] * kImageToMapScale;
    }
    image_coordinates.col(num_valid - 1) =
        all_peaks[object.center] * kImageToMapScale;
    output.push_back({std::move(valid_indices), std::move(image_coordinates)});
  }
  return output;
}

gxf::Expected<std::array<double, kExpectedPoseAsTensorSize>>
ExtractPose(const DopeObjectKeypoints &object,
            const Eigen::Matrix<double, 3, kNumVertexChannel> &cuboid_3d_points,
            const cv::Mat &camera_matrix) {
  const auto &valid_points = object.first;
  const size_t num_valid_points = valid_points.size();
  Eigen::Matrix3Xd keypoints_3d(3, num_valid_points);
  for (size_t j = 0; j < num_valid_points; ++j) {
    keypoints_3d.col(j) = cuboid_3d_points.col(valid_points[j]);
  }

  Pose3d pose;
  cv::Mat rvec, tvec;
  cv::Mat dist_coeffs = cv::Mat::zeros(1, 4, CV_64FC1); // no distortion

  cv::Mat cv_keypoints_3d;
  cv::eigen2cv(keypoints_3d, cv_keypoints_3d);
  cv::Mat cv_keypoints_2d;
  cv::eigen2cv(object.second, cv_keypoints_2d);
  if (!cv::solvePnP(cv_keypoints_3d.t(), cv_keypoints_2d.t(), camera_matrix,
                    dist_coeffs, rvec, tvec)) {
    GXF_LOG_ERROR("cv::solvePnP failed");
    return nvidia::gxf::Unexpected{GXF_FAILURE};
  }
  cv::cv2eigen(tvec, pose.translation);

  cv::Mat R;
  cv::Rodrigues(rvec, R); // R is 3x3
  Eigen::Matrix3d e_mat;
  cv::cv2eigen(R, e_mat);
  pose.rotation = Eigen::Quaterniond(e_mat);

  // If the Z coordinate is negative, the pose is placing the object behind
  // the camera (which is incorrect), so we flip it
  if (pose.translation[2] < 0.f) {
    pose = pose.inverse();
  }

  constexpr double kCentimeterToMeter = 100.0;
  // Return pose data as array
  return std::array<double, kExpectedPoseAsTensorSize>{
      pose.translation[0] / kCentimeterToMeter,
      pose.translation[1] / kCentimeterToMeter,
      pose.translation[2] / kCentimeterToMeter,
      pose.rotation.x(),
      pose.rotation.y(),
      pose.rotation.z(),
      pose.rotation.w()};
}

gxf::Expected<void> AddInputTimestampToOutput(gxf::Entity &output,
                                              gxf::Entity input) {
  std::string timestamp_name{"timestamp"};
  auto maybe_timestamp = input.get<gxf::Timestamp>(timestamp_name.c_str());

  // Default to unnamed
  if (!maybe_timestamp) {
    timestamp_name = std::string{""};
    maybe_timestamp = input.get<gxf::Timestamp>(timestamp_name.c_str());
  }

  if (!maybe_timestamp) {
    GXF_LOG_ERROR("Failed to get input timestamp!");
    return gxf::ForwardError(maybe_timestamp);
  }

  auto maybe_out_timestamp = output.add<gxf::Timestamp>(timestamp_name.c_str());
  if (!maybe_out_timestamp) {
    GXF_LOG_ERROR("Failed to add timestamp to output message!");
    return gxf::ForwardError(maybe_out_timestamp);
  }

  *maybe_out_timestamp.value() = *maybe_timestamp.value();
  return gxf::Success;
}

} // namespace

gxf_result_t
DopeDecoder::registerInterface(gxf::Registrar *registrar) noexcept {
  gxf::Expected<void> result;

  result &= registrar->parameter(tensorlist_receiver_, "tensorlist_receiver",
                                 "Tensorlist Input",
                                 "The detections as a tensorlist");

  result &= registrar->parameter(posearray_transmitter_,
                                 "posearray_transmitter", "PoseArray output",
                                 "The ouput poses as a pose array");

  result &= registrar->parameter(allocator_, "allocator", "Allocator",
                                 "Output Allocator");

  result &= registrar->parameter(
      object_dimensions_param_, "object_dimensions",
      "The dimensions of the object whose pose is being estimated");

  result &= registrar->parameter(
      camera_matrix_param_, "camera_matrix",
      "The parameters of the camera used to capture the input images");

  result &= registrar->parameter(
      object_name_, "object_name",
      "The class name of the object whose pose is being estimated");

  return gxf::ToResultCode(result);
}

gxf_result_t DopeDecoder::start() noexcept {
  // Extract 3D coordinates of bounding cuboid + centroid from object dimensions
  auto dims = object_dimensions_param_.get();
  if (dims.size() != 3) {
    GXF_LOG_ERROR("Expected object dimensions vector to be length 3 but got %lu",
                  dims.size());
    return GXF_FAILURE;
  }
  cuboid_3d_points_ = CuboidVertices({dims.at(0), dims.at(1), dims.at(2)});

  // Load camera matrix into cv::Mat
  if (camera_matrix_param_.get().size() != 9) {
    GXF_LOG_ERROR("Expected camera matrix vector to be length 9 but got %lu",
                  camera_matrix_param_.get().size());
    return GXF_FAILURE;
  }
  camera_matrix_ = cv::Mat::zeros(3, 3, CV_64FC1);
  std::memcpy(camera_matrix_.data, camera_matrix_param_.get().data(),
              camera_matrix_param_.get().size() * sizeof(double));

  return GXF_SUCCESS;
}

gxf_result_t DopeDecoder::tick() noexcept {
  const auto maybe_beliefmaps_message = tensorlist_receiver_->receive();
  if (!maybe_beliefmaps_message) {
    return gxf::ToResultCode(maybe_beliefmaps_message);
  }

  auto maybe_tensor = maybe_beliefmaps_message.value().get<gxf::Tensor>();
  if (!maybe_tensor) {
    return gxf::ToResultCode(maybe_tensor);
  }

  auto belief_maps = maybe_tensor.value();

  // Ensure belief maps match expected shape in first two dimensions
  if (belief_maps->shape().dimension(0) != kNumTensors ||
    belief_maps->shape().dimension(1) != kInputMapsChannels)
  {
    GXF_LOG_ERROR(
      "Belief maps had unexpected shape in first two dimensions: {%d, %d, %d, %d}",
      belief_maps->shape().dimension(0), belief_maps->shape().dimension(1),
      belief_maps->shape().dimension(2), belief_maps->shape().dimension(3));
    return GXF_FAILURE;
  }

  // Allocate output message
  auto maybe_posearray_message = gxf::Entity::New(context());
  if (!maybe_posearray_message) {
    GXF_LOG_ERROR("Failed to allocate PoseArray Message");
    return gxf::ToResultCode(maybe_posearray_message);
  }
  auto posearray_message = maybe_posearray_message.value();

  // Copy tensor data over to a more portable form
  std::array<cv::Mat, kInputMapsChannels> maps;
  const int input_map_row{belief_maps->shape().dimension(2)};
  const int input_map_column{belief_maps->shape().dimension(3)};
  for (size_t chan = 0; chan < kInputMapsChannels; ++chan) {
    maps[chan] = cv::Mat(input_map_row, input_map_column, CV_32F);
    const size_t stride = input_map_row * input_map_column * sizeof(float);

    const cudaMemcpyKind operation = cudaMemcpyDeviceToHost;
    const cudaError_t cuda_error =
        cudaMemcpy(maps[chan].data, belief_maps->pointer() + chan * stride,
                   stride, operation);

    if (cuda_error != cudaSuccess) {
      GXF_LOG_ERROR("Failed to copy data to Matrix: %s (%s)",
                    cudaGetErrorName(cuda_error),
                    cudaGetErrorString(cuda_error));
      return GXF_FAILURE;
    }
  }

  // Analyze the belief map to find vertex locations in image space
  const std::vector<DopeObjectKeypoints> dope_objects = FindObjects(maps);
  
  // Add timestamp to output msg
  auto maybe_added_timestamp = AddInputTimestampToOutput(
      posearray_message, maybe_beliefmaps_message.value());
  if (!maybe_added_timestamp) {
    return gxf::ToResultCode(maybe_added_timestamp);
  }

  if (dope_objects.empty()) {
    GXF_LOG_INFO("No objects detected.");

    return gxf::ToResultCode(
        posearray_transmitter_->publish(std::move(posearray_message)));
  }

  // Run Perspective-N-Point on the detected objects to find the 6-DoF pose of
  // the bounding cuboid
  for (const DopeObjectKeypoints &object : dope_objects) {
    // Allocate output tensor for this pose
    auto maybe_pose_tensor = posearray_message.add<gxf::Tensor>();
    if (!maybe_pose_tensor) {
      GXF_LOG_ERROR("Failed to allocate Pose Tensor");
      return gxf::ToResultCode(maybe_pose_tensor);
    }
    auto pose_tensor = maybe_pose_tensor.value();

    // Initializing GXF tensor
    auto result = pose_tensor->reshape<double>(
        nvidia::gxf::Shape{kExpectedPoseAsTensorSize},
        nvidia::gxf::MemoryStorageType::kDevice, allocator_);
    if (!result) {
      GXF_LOG_ERROR("Failed to reshape Pose Tensor to (%d,)",
                    kExpectedPoseAsTensorSize);
      return gxf::ToResultCode(result);
    }

    auto maybe_pose = ExtractPose(object, cuboid_3d_points_, camera_matrix_);
    if (!maybe_pose) {
      GXF_LOG_ERROR("Failed to extract pose from object");
      return gxf::ToResultCode(maybe_pose);
    }
    auto pose = maybe_pose.value();

    // Copy pose data into pose tensor
    const cudaMemcpyKind operation = cudaMemcpyHostToDevice;
    const cudaError_t cuda_error = cudaMemcpy(
        pose_tensor->pointer(), pose.data(), pose_tensor->size(), operation);

    if (cuda_error != cudaSuccess) {
      GXF_LOG_ERROR("Failed to copy data to pose tensor: %s (%s)",
                    cudaGetErrorName(cuda_error),
                    cudaGetErrorString(cuda_error));
      return GXF_FAILURE;
    }
  }

  return gxf::ToResultCode(
      posearray_transmitter_->publish(std::move(posearray_message)));
}

} // namespace dope
} // namespace isaac_ros
} // namespace nvidia
