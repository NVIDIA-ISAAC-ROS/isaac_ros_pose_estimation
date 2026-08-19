// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_dope/dope_decoder_node.hpp"

#include "opencv2/opencv.hpp"
#include "opencv2/core/eigen.hpp"

#if __GNUC__ < 9
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <string>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "isaac_ros_common/qos.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include "vision_msgs/msg/object_hypothesis_with_pose.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dope
{
namespace
{
// The dimensions of the input belief map tensor. The two major dimensions are fixed and
// determined by the output size of the DOPE DNN.
constexpr size_t kInputMapsChannels = 25;
constexpr size_t kNumTensors = 1;
constexpr char kCameraInfoTopicName[] = "camera_info";
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
struct DopeObject
{
  explicit DopeObject(int id)
  : center(id)
  {
    for (size_t ii = 0; ii < kNumCorners; ++ii) {
      corners[ii] = kInvalidId;
      best_distances[ii] = kInvalidDist;
    }
  }

  int center;
  std::array<int, kNumCorners> corners;
  std::array<float, kNumCorners> best_distances;
};

struct Pose3d
{
public:
  Eigen::Vector3d translation;
  Eigen::Quaterniond rotation;

  Pose3d inverse()
  {
    Pose3d retval;
    retval.translation = -this->translation;
    retval.rotation = this->rotation.inverse();
    return retval;
  }
};

constexpr char INPUT_TOPIC_NAME[] = "belief_map_array";
constexpr char OUTPUT_TOPIC_NAME[] = "dope/detections";

// Returns pixel mask for local maximums in single - channel image src
void IsolateMaxima(const cv::Mat & src, cv::Mat & mask)
{
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
std::vector<Eigen::Vector2i> FindPeaks(const cv::Mat & image)
{
  // Extract centers of local maxima
  cv::Mat mask;
  std::vector<cv::Point> maxima;
  IsolateMaxima(image, mask);
  cv::findNonZero(mask, maxima);

  // Find maxima
  std::vector<Eigen::Vector2i> peaks;
  for (const auto & m : maxima) {
    if (image.at<float>(m.y, m.x) > kBlurredPeakThreshold) {
      peaks.push_back(Eigen::Vector2i(m.x, m.y));
    }
  }

  return peaks;
}

// Returns 3x9 matrix of the 3d coordinates of cuboid corners and center
Eigen::Matrix<double, 3, kNumVertexChannel>
CuboidVertices(const std::array<double, 3> & ext)
{
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
FindObjects(
  const std::array<cv::Mat, kInputMapsChannels> & maps, const double map_peak_threshold,
  const double affinity_map_angle_threshold)
{
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
            row >= image.size[0])
          {
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
    const std::vector<int> & peaks = channel_peaks[chan];
    for (size_t pp = 0; pp < peaks.size(); ++pp) {
      int best_idx = kInvalidId;
      float best_distance = kInvalidDist;
      float best_angle = kInvalidDist;

      for (size_t jj = 0; jj < objects.size(); ++jj) {
        const Vector2f & center = all_peaks[objects[jj].center];
        const Vector2f & point = all_peaks[peaks[pp]];
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
        best_distance < objects[best_idx].best_distances[chan]))
      {
        objects[best_idx].corners[chan] = peaks[pp];
        objects[best_idx].best_distances[chan] = best_distance;
      }
    }
  }

  std::vector<DopeObjectKeypoints> output;
  for (const DopeObject & object : objects) {
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


std::array<double, kExpectedPoseAsTensorSize>
ExtractPose(
  const DopeObjectKeypoints & object,
  const Eigen::Matrix<double, 3, kNumVertexChannel> & cuboid_3d_points,
  const cv::Mat & camera_matrix, const double rotation_y_axis,
  const double rotation_x_axis, const double rotation_z_axis)
{
  const auto & valid_points = object.first;
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
                    dist_coeffs, rvec, tvec, false, cv::SOLVEPNP_EPNP))
  {
    RCLCPP_ERROR(rclcpp::get_logger("dope_decoder_node"), "cv::solvePnP failed");
    return {};
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

}  // namespace

DopeDecoderNode::DopeDecoderNode(const rclcpp::NodeOptions & options)
:  rclcpp::Node("dope_decoder_node", options),
  // Parameters
  configuration_file_(declare_parameter<std::string>("configuration_file", "dope_config.yaml")),
  object_name_(declare_parameter<std::string>("object_name", "Ketchup")),
  tf_frame_name_(declare_parameter<std::string>("tf_frame_name", "dope_object")),
  enable_tf_publishing_(declare_parameter<bool>("enable_tf_publishing", true)),
  map_peak_threshold_(declare_parameter<double>("map_peak_threshold", 0.1)),
  affinity_map_angle_threshold_(declare_parameter<double>("affinity_map_angle_threshold", 0.5)),
  rotation_y_axis_(declare_parameter<double>("rotation_y_axis", false)),
  rotation_x_axis_(declare_parameter<double>("rotation_x_axis", false)),
  rotation_z_axis_(declare_parameter<double>("rotation_z_axis", false)),
  object_dimensions_{},
  camera_matrix_{},
  input_qos_(::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")),
  output_qos_(::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")),
  input_queue_size_(declare_parameter<int16_t>("input_queue_size", 10)),
  tensor_sub_{},
  camera_info_sub_{},
  exact_sync_{
    ExactPolicy(static_cast<uint32_t>(input_queue_size_)), tensor_sub_, camera_info_sub_}
{
  RCLCPP_DEBUG(get_logger(), "[DopeDecoderNode] Constructor");

  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("DopeDecoderNode");


  // Add callback function for Dope Pose Array to broadcast to ROS TF tree if setting is enabled.
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  exact_sync_.registerCallback(
    std::bind(
      &DopeDecoderNode::DopeDecoderDetectionCallback, this,
      std::placeholders::_1, std::placeholders::_2));
  const auto input_qos_profile = input_qos_.get_rmw_qos_profile();
  tensor_sub_.subscribe(this, INPUT_TOPIC_NAME, input_qos_profile, sub_options);
  camera_info_sub_.subscribe(this, kCameraInfoTopicName, input_qos_profile, sub_options);

  detections_pub_ = create_publisher<vision_msgs::msg::Detection3DArray>(
    OUTPUT_TOPIC_NAME, output_qos_, pub_options);
  // Open configuration YAML file
  const std::string package_directory = ament_index_cpp::get_package_share_directory(
    "isaac_ros_dope");
  fs::path yaml_path = package_directory / fs::path("config") / fs::path(configuration_file_);
  if (!fs::exists(yaml_path)) {
    RCLCPP_ERROR(this->get_logger(), "%s could not be found. Exiting.", yaml_path.string().c_str());
    throw std::runtime_error("Parameter parsing failure.");
  }

  // Parse parameters
  rcl_params_t * dope_params = rcl_yaml_node_struct_init(rcutils_get_default_allocator());
  rcl_parse_yaml_file(yaml_path.c_str(), dope_params);

  const std::string dimensions_param = "dimensions." + object_name_;
  rcl_variant_t * dimensions =
    rcl_yaml_node_struct_get("dope", dimensions_param.c_str(), dope_params);
  if (!dimensions->double_array_value) {
    RCLCPP_ERROR(
      this->get_logger(), "No dimensions parameter found for object name: %s",
      object_name_.c_str());
    throw std::runtime_error("Parameter parsing failure.");
  }

  auto dd = dimensions->double_array_value->values;
  object_dimensions_ = {dd[0], dd[1], dd[2]};

  cuboid_3d_points_ = CuboidVertices(
    {object_dimensions_.at(0), object_dimensions_.at(1), object_dimensions_.at(2)});

  cv_camera_matrix_ = cv::Mat::zeros(3, 3, CV_64FC1);
}

DopeDecoderNode::~DopeDecoderNode() {}

bool DopeDecoderNode::UpdateCameraProperties(
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info)
{
  if (!camera_info ||
    !std::isfinite(camera_info->k[0]) || camera_info->k[0] <= 0.0 ||
    !std::isfinite(camera_info->k[2]) ||
    !std::isfinite(camera_info->k[4]) || camera_info->k[4] <= 0.0 ||
    !std::isfinite(camera_info->k[5]))
  {
    return false;
  }

  cv_camera_matrix_.at<double>(0, 0) = camera_info->k[0];
  cv_camera_matrix_.at<double>(0, 2) = camera_info->k[2];
  cv_camera_matrix_.at<double>(1, 1) = camera_info->k[4];
  cv_camera_matrix_.at<double>(1, 2) = camera_info->k[5];
  cv_camera_matrix_.at<double>(2, 2) = 1.0;
  return true;
}
// convert Detection3DArray to ROS message that will be published to the TF tree.

void DopeDecoderNode::DopeDecoderDetectionCallback(
  const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & tensor_list,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info)
{
  if (!UpdateCameraProperties(camera_info)) {
    RCLCPP_ERROR(get_logger(), "CameraInfo has invalid intrinsics");
    return;
  }

  auto belief_maps = tensor_list->get_tensor(0);
  if (belief_maps.data_type() != nvidia::isaac_ros::nitros::NitrosDataType::kFloat32) {
    RCLCPP_ERROR(get_logger(), "Belief maps tensor has wrong type (expected float32)");
    throw std::runtime_error("Invalid belief maps tensor type");
  }

  const nvidia::isaac_ros::nitros::ReadHandle handle = belief_maps.get_read_handle(*cuda_stream_);
  // Ensure belief maps match expected shape in first two dimensions
  const nvidia::isaac_ros::nitros::NitrosTensorShape belief_maps_shape = belief_maps.shape();
  const auto & belief_maps_dims = belief_maps_shape.dims();
  if (belief_maps_dims.size() < 4 ||
    belief_maps_dims.at(0) != static_cast<int32_t>(kNumTensors) ||
    belief_maps_dims.at(1) != static_cast<int32_t>(kInputMapsChannels))
  {
    RCLCPP_ERROR(
      get_logger(), "Belief maps had unexpected shape in first two dimensions: {%d, %d, %d, %d}",
      belief_maps_dims.size() > 0 ? belief_maps_dims.at(0) : -1,
      belief_maps_dims.size() > 1 ? belief_maps_dims.at(1) : -1,
      belief_maps_dims.size() > 2 ? belief_maps_dims.at(2) : -1,
      belief_maps_dims.size() > 3 ? belief_maps_dims.at(3) : -1);
    throw std::runtime_error("Invalid belief maps shape");
  }
  // Copy tensor data over to a more portable form
  std::array<cv::Mat, kInputMapsChannels> maps;
  const int input_map_row{belief_maps_dims.at(2)};
  const int input_map_column{belief_maps_dims.at(3)};
  for (size_t chan = 0; chan < kInputMapsChannels; ++chan) {
    maps[chan] = cv::Mat(input_map_row, input_map_column, CV_32F);
    const size_t stride = input_map_row * input_map_column * sizeof(float);

    const cudaError_t cuda_error =
      cudaMemcpyAsync(maps[chan].data, handle.get_ptr() + chan * stride,
                  stride, cudaMemcpyDeviceToHost, *cuda_stream_);
    CHECK_CUDA_ERROR(cuda_error, "Failed to copy data to Matrix");
  }
  auto cuda_sync_error = cudaStreamSynchronize(*cuda_stream_);
  CHECK_CUDA_ERROR(cuda_sync_error, "Failed to synchronize stream");

  // Analyze the belief map to find vertex locations in image space
  const std::vector<DopeObjectKeypoints> dope_objects =
    FindObjects(maps, map_peak_threshold_, affinity_map_angle_threshold_);

  // convert from NitrosTensorList to Detection3DArray message
  vision_msgs::msg::Detection3DArray detection3darray_message;
  detection3darray_message.header = camera_info->header;
  if (dope_objects.empty()) {
    RCLCPP_DEBUG(get_logger(), "No objects detected.");
  }
  geometry_msgs::msg::TransformStamped transform_stamped;
  int child_frame_id_num = 1;
  for (size_t i = 0; i < dope_objects.size(); ++i) {
    const std::array<double, kExpectedPoseAsTensorSize> pose = ExtractPose(
      dope_objects.at(i), cuboid_3d_points_, cv_camera_matrix_,
      rotation_y_axis_, rotation_x_axis_, rotation_z_axis_);
    const bool pose_is_finite =
      std::all_of(pose.cbegin(), pose.cend(), [](double value) {return std::isfinite(value);});
    const double q_norm_sq =
      pose[3] * pose[3] + pose[4] * pose[4] + pose[5] * pose[5] + pose[6] * pose[6];
    if (!pose_is_finite || q_norm_sq < 1e-20) {
      RCLCPP_ERROR(get_logger(), "Failed to extract pose from object");
      continue;
    }
    vision_msgs::msg::Detection3D detection3d_message;
    detection3d_message.header = detection3darray_message.header;
    detection3d_message.bbox.size.x = object_dimensions_.at(0) / kCentimeterToMeter;
    detection3d_message.bbox.size.y = object_dimensions_.at(1) / kCentimeterToMeter;
    detection3d_message.bbox.size.z = object_dimensions_.at(2) / kCentimeterToMeter;
    detection3d_message.bbox.center.position.x = pose[0];
    detection3d_message.bbox.center.position.y = pose[1];
    detection3d_message.bbox.center.position.z = pose[2];
    detection3d_message.bbox.center.orientation.x = pose[3];
    detection3d_message.bbox.center.orientation.y = pose[4];
    detection3d_message.bbox.center.orientation.z = pose[5];
    detection3d_message.bbox.center.orientation.w = pose[6];
    vision_msgs::msg::ObjectHypothesisWithPose object_hypothesis_message;
    object_hypothesis_message.hypothesis.class_id = object_name_;
    object_hypothesis_message.hypothesis.score = 0.0;
    object_hypothesis_message.pose.pose = detection3d_message.bbox.center;
    detection3d_message.results.push_back(object_hypothesis_message);
    detection3darray_message.detections.push_back(detection3d_message);

    if (enable_tf_publishing_) {
      transform_stamped.header.stamp = now();
      transform_stamped.header.frame_id = tensor_list->get_frame_id();
      transform_stamped.child_frame_id = tf_frame_name_ + std::to_string(child_frame_id_num);
      // ExtractPose: translation (m) then quaternion xyzw.
      transform_stamped.transform.translation.x = pose[0];
      transform_stamped.transform.translation.y = pose[1];
      transform_stamped.transform.translation.z = pose[2];
      transform_stamped.transform.rotation.x = pose[3];
      transform_stamped.transform.rotation.y = pose[4];
      transform_stamped.transform.rotation.z = pose[5];
      transform_stamped.transform.rotation.w = pose[6];

      tf_broadcaster_->sendTransform(transform_stamped);
    }
    child_frame_id_num++;
  }
  detections_pub_->publish(detection3darray_message);
}

}  // namespace dope
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dope::DopeDecoderNode)
