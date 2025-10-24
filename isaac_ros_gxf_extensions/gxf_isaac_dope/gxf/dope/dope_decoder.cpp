// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime.h>
#include <Eigen/Dense>

#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "detection3_d_array_message/detection3_d_array_message.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgproc.hpp"

#include "gxf/core/parameter_parser_std.hpp"
#include "gxf/multimedia/camera.hpp"
#include "gxf/multimedia/video.hpp"
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
// Offset added to belief map pixel coordinate, constant for the fixed input
// image size
// https://github.com/NVlabs/Deep_Object_Pose/blob/master/src/dope/inference/detector.py
// line 343
constexpr float kOffsetDueToUpsampling = 0.4395f;
// Minimum required blurred belief map value at the peaks
constexpr float kBlurredPeakThreshold = 0.01;
// The original image is kImageToMapScale larger in each dimension than the
// output tensor from the DNN.
constexpr float kImageToMapScale = 8.0f;
// Require all 9 vertices to publish a pose
constexpr size_t kRequiredPointsForPnP = 9;
// Placeholder for unidentify peak ids in DopeObject
constexpr int kInvalidId = -1;
// Placeholder for unknown best distance from centroid to peak in DopeObject
constexpr float kInvalidDist = std::numeric_limits<float>::max();
// For converting sizes in cm to m in ExtractPose and when publishing bbox sizes.
constexpr double kCentimeterToMeter = 100.0;

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
void IsolateMaxima(const cv::Mat& src, cv::Mat& mask) {
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
std::vector<Eigen::Vector2i> FindPeaks(const cv::Mat& image) {
  // Extract centers of local maxima
  cv::Mat mask;
  std::vector<cv::Point> maxima;
  IsolateMaxima(image, mask);
  cv::findNonZero(mask, maxima);

  // Find maxima
  std::vector<Eigen::Vector2i> peaks;
  for (const auto &m : maxima) {
    if (image.at<float>(m.y, m.x) > kBlurredPeakThreshold) {
      peaks.push_back(Eigen::Vector2i(m.x, m.y));
    }
  }

  return peaks;
}

// Returns 3x9 matrix of the 3d coordinates of cuboid corners and center
Eigen::Matrix<double, 3, kNumVertexChannel>
CuboidVertices(const std::array<double, 3>& ext) {
  // X axis points to the right
  const double right = -ext.at(0) * 0.5;
  const double left = ext.at(0) * 0.5;
  // Y axis points downward
  const double bottom = -ext.at(1) * 0.5;
  const double top = ext.at(1) * 0.5;
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
FindObjects(const std::array<cv::Mat, kInputMapsChannels>& maps, const double map_peak_threshold,
            const double affinity_map_angle_threshold) {
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
    std::vector<Vector2i> peaks = FindPeaks(blurred);
    for (size_t pp = 0; pp < peaks.size(); ++pp) {
      const auto peak = peaks[pp];

      // Compute the weighted average for localizing the peak, using an 11x11
      // window
      Vector2f peak_sum(0, 0);
      float weight_sum = 0.0f;
      for (int ii = -5; ii <= 5; ++ii) {
        for (int jj = -5; jj <= 5; ++jj) {
          const int row = peak[1] + ii;
          const int col = peak[0] + jj;

          if (col < 0 || col >= image.size[1] || row < 0 ||
              row >= image.size[0]) {
            continue;
          }

          const float weight = image.at<float>(row, col);
          weight_sum += weight;
          peak_sum[1] += row * weight;
          peak_sum[0] += col * weight;
        }
      }

      if (image.at<float>(peak[1], peak[0]) >= map_peak_threshold) {
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
    const std::vector<int>& peaks = channel_peaks[chan];
    for (size_t pp = 0; pp < peaks.size(); ++pp) {
      int best_idx = kInvalidId;
      float best_distance = kInvalidDist;
      float best_angle = kInvalidDist;

      for (size_t jj = 0; jj < objects.size(); ++jj) {
        const Vector2f& center = all_peaks[objects[jj].center];
        const Vector2f& point = all_peaks[peaks[pp]];
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

        if (angle < affinity_map_angle_threshold && dist < best_distance) {
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
          (best_angle < affinity_map_angle_threshold &&
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
ExtractPose(const DopeObjectKeypoints& object,
            const Eigen::Matrix<double, 3, kNumVertexChannel>& cuboid_3d_points,
            const cv::Mat& camera_matrix, const double rotation_y_axis,
            const double rotation_x_axis, const double rotation_z_axis) {
  const auto& valid_points = object.first;
  const size_t num_valid_points = valid_points.size();
  Eigen::Matrix3Xd keypoints_3d(3, num_valid_points);
  for (size_t j = 0; j < num_valid_points; ++j) {
    keypoints_3d.col(j) = cuboid_3d_points.col(valid_points[j]);
  }

  Pose3d pose;
  cv::Mat rvec, tvec;
  cv::Mat dist_coeffs = cv::Mat::zeros(1, 4, CV_64FC1);  // no distortion

  cv::Mat cv_keypoints_3d;
  cv::eigen2cv(keypoints_3d, cv_keypoints_3d);
  cv::Mat cv_keypoints_2d;
  cv::eigen2cv(object.second, cv_keypoints_2d);
  if (!cv::solvePnP(cv_keypoints_3d.t(), cv_keypoints_2d.t(), camera_matrix,
                    dist_coeffs, rvec, tvec, false, cv::SOLVEPNP_EPNP)) {
    GXF_LOG_ERROR("cv::solvePnP failed");
    return nvidia::gxf::Unexpected{GXF_FAILURE};
  }
  cv::cv2eigen(tvec, pose.translation);

  cv::Mat R;
  cv::Rodrigues(rvec, R);  // R is 3x3
  Eigen::Matrix3d e_mat;
  cv::cv2eigen(R, e_mat);
  Eigen::AngleAxisd rotation_y(rotation_y_axis, Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd rotation_x(rotation_x_axis, Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd rotation_z(rotation_z_axis, Eigen::Vector3d::UnitZ());

  // Convert to rotation matrices
  Eigen::Matrix3d rotation_y_matrix = rotation_y.toRotationMatrix();
  Eigen::Matrix3d rotation_x_matrix = rotation_x.toRotationMatrix();
  Eigen::Matrix3d rotation_z_matrix = rotation_z.toRotationMatrix();

  // Compose the rotations
  Eigen::Matrix3d composed_rotation = rotation_z_matrix * rotation_y_matrix * rotation_x_matrix;

  // Apply the composed rotation to the original matrix
  Eigen::Matrix3d rotated_matrix = e_mat * composed_rotation;

  pose.rotation = Eigen::Quaterniond(rotated_matrix);


  // If the Z coordinate is negative, the pose is placing the object behind
  // the camera (which is incorrect), so we flip it
  if (pose.translation[2] < 0.f) {
    pose = pose.inverse();
  }

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

gxf::Expected<void> AddInputTimestampToOutput(gxf::Entity& output, gxf::Entity input) {
  std::string named_timestamp{"timestamp"};
  std::string unnamed_timestamp{""};
  auto maybe_input_timestamp = input.get<gxf::Timestamp>(named_timestamp.c_str());

  // Try to get a named timestamp from the input entity
  if (!maybe_input_timestamp) {
    maybe_input_timestamp = input.get<gxf::Timestamp>(unnamed_timestamp.c_str());
  }
  // If there is no named timestamp, try to get a unnamed timestamp from the input entity
  if (!maybe_input_timestamp) {
    GXF_LOG_ERROR("Failed to get input timestamp!");
    return gxf::ForwardError(maybe_input_timestamp);
  }

  // Try to get a named timestamp from the output entity
  auto maybe_output_timestamp = output.get<gxf::Timestamp>(named_timestamp.c_str());
  // If there is no named timestamp, try to get a unnamed timestamp from the output entity
  if (!maybe_output_timestamp) {
    maybe_output_timestamp = output.get<gxf::Timestamp>(unnamed_timestamp.c_str());
  }

  // If there is no unnamed timestamp also, then add a named timestamp to the output entity
  if (!maybe_output_timestamp) {
    maybe_output_timestamp = output.add<gxf::Timestamp>(named_timestamp.c_str());
    if (!maybe_output_timestamp) {
      GXF_LOG_ERROR("Failed to add timestamp to output message!");
      return gxf::ForwardError(maybe_output_timestamp);
    }
  }

  *maybe_output_timestamp.value() = *maybe_input_timestamp.value();
  return gxf::Success;
}

}  // namespace

gxf_result_t
DopeDecoder::registerInterface(gxf::Registrar* registrar) noexcept {
  gxf::Expected<void> result;

  result &= registrar->parameter(tensorlist_receiver_, "tensorlist_receiver",
                                 "Tensorlist Input",
                                 "The detections as a tensorlist");

  result &= registrar->parameter(camera_model_input_, "camera_model_input",
                                  "Camera Model Input",
                                  "The Camera intrinsics as a Nitros Camera Info type");

  result &= registrar->parameter(detection3darray_transmitter_,
                                 "detection3darray_transmitter", "Detection3DArray output",
                                 "The ouput poses as a Detection3D array");

  result &= registrar->parameter(allocator_, "allocator", "Allocator",
                                 "Output Allocator");

  result &= registrar->parameter(
      object_dimensions_param_, "object_dimensions",
      "The dimensions of the object whose pose is being estimated");

  result &= registrar->parameter(
      object_name_, "object_name",
      "The class name of the object whose pose is being estimated");

  result &= registrar->parameter(
      map_peak_threshold_, "map_peak_threshold",
      "The minimum value of a peak in a belief map");

  result &= registrar->parameter(
      affinity_map_angle_threshold_, "affinity_map_angle_threshold",
      "The maximum angle threshold for affinity mapping of corners to centroid");

  result &= registrar->parameter(
      rotation_y_axis_, "rotation_y_axis", "rotation_y_axis",
      "Rotate Dope pose by N degrees along y axis", 0.0);

  result &= registrar->parameter(
      rotation_x_axis_, "rotation_x_axis", "rotation_x_axis",
      "Rotate Dope pose by N degrees along x axis", 0.0);

  result &= registrar->parameter(
      rotation_z_axis_, "rotation_z_axis", "rotation_z_axis",
      "Rotate Dope pose by N degrees along z axis", 0.0);

  result &= registrar->parameter(cuda_stream_pool_, "stream_pool", "Cuda Stream Pool",
                                 "Instance of gxf::CudaStreamPool to allocate CUDA stream.");

  return gxf::ToResultCode(result);
}

gxf_result_t DopeDecoder::start() noexcept {
  // Get cuda stream from stream pool
  auto maybe_stream = cuda_stream_pool_.get()->allocateStream();
  if (!maybe_stream) { return gxf::ToResultCode(maybe_stream); }

  cuda_stream_handle_ = std::move(maybe_stream.value());
  if (!cuda_stream_handle_->stream()) {
    GXF_LOG_ERROR("Allocated stream is not initialized!");
    return GXF_FAILURE;
  }
  if (!cuda_stream_handle_.is_null()) {
    cuda_stream_ = cuda_stream_handle_->stream().value();
  }

  // Extract 3D coordinates of bounding cuboid + centroid from object dimensions
  auto dims = object_dimensions_param_.get();
  if (dims.size() != 3) {
    return GXF_FAILURE;
  }
  cuboid_3d_points_ = CuboidVertices({dims.at(0), dims.at(1), dims.at(2)});

  camera_matrix_ = cv::Mat::zeros(3, 3, CV_64FC1);

  return GXF_SUCCESS;
}

gxf::Expected<void> DopeDecoder::updateCameraProperties(
    gxf::Handle<gxf::CameraModel> camera_model) {
  if (!camera_model) {
    return gxf::Unexpected{GXF_FAILURE};
  }

  camera_matrix_.at<double>(0, 0) = camera_model->focal_length.x;
  camera_matrix_.at<double>(0, 2)= camera_model->principal_point.x;
  camera_matrix_.at<double>(1, 1) = camera_model->focal_length.y;
  camera_matrix_.at<double>(1, 2) = camera_model->principal_point.y;
  camera_matrix_.at<double>(2, 2) = 1.0;

  return gxf::Success;
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
    belief_maps->shape().dimension(1) != kInputMapsChannels) {
    GXF_LOG_ERROR(
      "Belief maps had unexpected shape in first two dimensions: {%d, %d, %d, %d}",
      belief_maps->shape().dimension(0), belief_maps->shape().dimension(1),
      belief_maps->shape().dimension(2), belief_maps->shape().dimension(3));
    return GXF_FAILURE;
  }
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

  // Copy tensor data over to a more portable form
  std::array<cv::Mat, kInputMapsChannels> maps;
  const int input_map_row{belief_maps->shape().dimension(2)};
  const int input_map_column{belief_maps->shape().dimension(3)};
  for (size_t chan = 0; chan < kInputMapsChannels; ++chan) {
    maps[chan] = cv::Mat(input_map_row, input_map_column, CV_32F);
    const size_t stride = input_map_row * input_map_column * sizeof(float);

    const cudaMemcpyKind operation = cudaMemcpyDeviceToHost;
    const cudaError_t cuda_error =
        cudaMemcpyAsync(maps[chan].data, belief_maps->pointer() + chan * stride,
                   stride, operation, cuda_stream_);

    if (cuda_error != cudaSuccess) {
      GXF_LOG_ERROR("Failed to copy data to Matrix: %s (%s)",
                    cudaGetErrorName(cuda_error),
                    cudaGetErrorString(cuda_error));
      return GXF_FAILURE;
    }

    auto cuda_sync_error = cudaStreamSynchronize(cuda_stream_);
    if (cuda_sync_error != cudaSuccess) {
      GXF_LOG_ERROR("Failed to synchronize stream: %s (%s)",
                    cudaGetErrorName(cuda_sync_error),
                    cudaGetErrorString(cuda_sync_error));
      return GXF_FAILURE;
    }
  }

  // Analyze the belief map to find vertex locations in image space
  const std::vector<DopeObjectKeypoints> dope_objects =
      FindObjects(maps, map_peak_threshold_, affinity_map_angle_threshold_);

  // Create Detection3DList Message
  auto maybe_detection3_d_list = nvidia::isaac::CreateDetection3DListMessage(
      context(), dope_objects.size());
  if (!maybe_detection3_d_list) {
    GXF_LOG_ERROR("[DopeDecoder] Failed to create detection3d list");
    return gxf::ToResultCode(maybe_detection3_d_list);
  }
  auto detection3_d_list = maybe_detection3_d_list.value();

  auto maybe_added_timestamp =
      AddInputTimestampToOutput(maybe_detection3_d_list->entity, maybe_beliefmaps_message.value());
  if (!maybe_added_timestamp) {
    GXF_LOG_ERROR("[DopeDecoder] Failed to add timestamp");
  }

  if (dope_objects.empty()) {
    GXF_LOG_INFO("No objects detected.");
    return gxf::ToResultCode(
        detection3darray_transmitter_->publish(std::move(detection3_d_list.entity)));
  }

  // Run Perspective-N-Point on the detected objects to find the 6-DoF pose of
  // the bounding cuboid
  for (size_t i = 0; i < dope_objects.size(); i++) {
    const DopeObjectKeypoints& object = dope_objects[i];
    auto maybe_pose = ExtractPose(object, cuboid_3d_points_, camera_matrix_,
                                  rotation_y_axis_.get(), rotation_x_axis_.get(),
                                  rotation_z_axis_.get());
    if (!maybe_pose) {
      GXF_LOG_ERROR("Failed to extract pose from object");
      return gxf::ToResultCode(maybe_pose);
    }
    auto pose = maybe_pose.value();

    // Rotation: w = pose[6], x = pose[3], y = pose[4], z = pose[5]
    nvidia::isaac::Pose3d pose3d{
      nvidia::isaac::SO3d::FromQuaternion({pose[6], pose[3], pose[4], pose[5]}),  // rotation
      nvidia::isaac::Vector3d(pose[0], pose[1], pose[2])  // translation
    };

    auto bbox_dims = object_dimensions_param_.get();
    auto bbox_x = bbox_dims.at(0) / kCentimeterToMeter;
    auto bbox_y = bbox_dims.at(1) / kCentimeterToMeter;
    auto bbox_z = bbox_dims.at(2) / kCentimeterToMeter;

    **detection3_d_list.poses[i] = pose3d;
    **detection3_d_list.bbox_sizes[i] = nvidia::isaac::Vector3f(bbox_x, bbox_y, bbox_z);
    **detection3_d_list.hypothesis[i] = nvidia::isaac::ObjectHypothesis{
        std::vector<float>{0.0}, std::vector<std::string>{object_name_}};
  }
  return gxf::ToResultCode(
      detection3darray_transmitter_->publish(std::move(detection3_d_list.entity)));
}

}  // namespace dope
}  // namespace isaac_ros
}  // namespace nvidia
