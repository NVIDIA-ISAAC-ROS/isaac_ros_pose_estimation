/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "isaac_ros_dope/dope_decoder_node.hpp"

#include <Eigen/Dense>
#include <sys/stat.h>

#include <algorithm>

#if __GNUC__ < 9
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "rcl_yaml_param_parser/parser.h"


namespace
{
// The dimensions of the input belief map tensor. This tensor size is fixed and determined by the
// output size of the DOPE DNN.
constexpr size_t kInputMapsRow = 60;
constexpr size_t kInputMapsColumn = 80;
constexpr size_t kInputMapsChannels = 25;
constexpr size_t kNumTensors = 1;
// The number of vertex (belief map) channels in the DNN output tensor for the 8
// corners and 1 centroid. The other channels are affinity maps (vector fields) for the 8 corners.
constexpr size_t kNumVertexChannel = 9;
// The standard deviation of the Gaussian blur
constexpr size_t kGaussianSigma = 3.0;
// Minimum acceptable sum of averaging weights
constexpr float kMinimumWeightSum = 1e-6;
constexpr float kMapPeakThreshhold = 0.01;
constexpr float kPeakWeightThreshhold = 0.1;
constexpr float kAffinityMapAngleThreshhold = 0.5;
// Offset added to belief map pixel coordinate, constant for the fixed input image size
// https://github.com/NVlabs/Deep_Object_Pose/blob/master/src/dope/inference/detector.py line 343
constexpr float kOffsetDueToUpsampling = 0.4395f;
// The original image is kImageToMapScale larger in each dimension than the output tensor from
// the DNN.
constexpr float kImageToMapScale = 8.0f;
// From the requirements listed for pnp::ComputeCameraPoseEpnp.
constexpr size_t kRequiredPointsForPnP = 6;
// Placeholder for unidentify peak ids in DopeObject
constexpr int kInvalidId = -1;
// Placeholder for unknown best distance from centroid to peak in DopeObject
constexpr float kInvalidDist = std::numeric_limits<float>::max();


// The list of keypoint indices (0-8, 8 is centroid) and their corresponding 2d pixel coordinates
// as columns in a matrix.
using DopeObjectKeypoints = std::pair<std::vector<int>, Eigen::Matrix2Xf>;

// An internal class used to store information about detected objects. Used only within the
// 'FindObjects' function.
struct DopeObject
{
  explicit DopeObject(int id)
  : center(id)
  {
    for (int ii = 0; ii < 8; ++ii) {
      corners[ii] = kInvalidId;
      best_distances[ii] = kInvalidDist;
    }
  }

  int center;
  std::array<int, 8> corners;
  std::array<float, 8> best_distances;
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

// Returns pixel mask for local maximums in single-channel image src
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
std::vector<Eigen::Vector2i> FindPeaks(const cv::Mat & image, const float threshold)
{
  // Extract centers of local maxima
  cv::Mat mask;
  std::vector<cv::Point> maxima;
  IsolateMaxima(image, mask);
  cv::findNonZero(mask, maxima);

  // Find maxima
  std::vector<Eigen::Vector2i> peaks;
  for (size_t ii = 0; ii < maxima.size(); ++ii) {
    if (image.at<float>(maxima[ii].y, maxima[ii].x) > threshold) {
      peaks.push_back(Eigen::Vector2i(maxima[ii].x, maxima[ii].y));
    }
  }

  return peaks;
}

// Returns 3x9 matrix of the 3d coordinates of cuboid corners and center
Eigen::Matrix<double, 3, kNumVertexChannel> CuboidVertices(const Eigen::Vector3d & ext)
{
  // X axis points to the right
  const double right = ext[0] * 0.5;
  const double left = -ext[0] * 0.5;
  // Y axis points downward
  const double bottom = ext[1] * 0.5;
  const double top = -ext[1] * 0.5;
  // Z axis points forward (away from camera)
  const double front = ext[2] * 0.5;
  const double rear = -ext[2] * 0.5;

  Eigen::Matrix<double, 3, kNumVertexChannel> points;
  points << right, left, left, right, right, left, left, right, 0.0,
    top, top, bottom, bottom, top, top, bottom, bottom, 0.0,
    front, front, front, front, rear, rear, rear, rear, 0.0;

  return points;
}

// Converts a vector of Poses to a shared pointer to a message
std::shared_ptr<geometry_msgs::msg::PoseArray> FormatPoseArray(
  const isaac_ros_nvengine_interfaces::msg::TensorList::ConstSharedPtr & belief_maps_msg,
  const std::vector<geometry_msgs::msg::Pose> & poses)
{
  auto pose_array_msg = std::make_shared<geometry_msgs::msg::PoseArray>();
  pose_array_msg->header = belief_maps_msg->header;
  pose_array_msg->poses = poses;
  return pose_array_msg;
}

}  // namespace


namespace isaac_ros
{
namespace dope
{
struct DopeDecoderNode::DopeDecoderImpl
{
  // The name of the logger that node will print to
  std::string logger_name_;
  Eigen::Vector3d cuboid_dimensions_;
  std::array<double, 9> camera_matrix_;

  explicit DopeDecoderImpl(DopeDecoderNode & node)
  : logger_name_(std::string(node.get_logger().get_name()))
  {
  }

  ~DopeDecoderImpl() = default;

  void Initialize(std::array<double, 3> & cuboid_dimensions, std::array<double, 9> & camera_matrix)
  {
    cuboid_dimensions_ << cuboid_dimensions[0], cuboid_dimensions[1], cuboid_dimensions[2];
    camera_matrix_ = camera_matrix;
  }

  std::vector<DopeObjectKeypoints> findObjects(const std::array<cv::Mat, kInputMapsChannels> & maps)
  {
    using Vector2f = Eigen::Vector2f;
    using Vector2i = Eigen::Vector2i;

    // 'all_peaks' contains: x,y: 2d location of peak; z: belief map value
    std::vector<Vector2f> all_peaks;
    std::array<std::vector<int>, kNumVertexChannel> channel_peaks;
    cv::Mat image = cv::Mat(kInputMapsRow, kInputMapsColumn, CV_32F);
    for (size_t chan = 0; chan < kNumVertexChannel; ++chan) {
      // Isolate and copy a single channel
      image = maps[chan].clone();

      // Smooth the image
      cv::Mat blurred;
      cv::GaussianBlur(
        image, blurred, cv::Size(0, 0), kGaussianSigma, kGaussianSigma,
        cv::BORDER_REFLECT);

      // Find the maxima of the tensor values in this channel
      std::vector<Vector2i> peaks = FindPeaks(blurred, kMapPeakThreshhold);
      for (size_t pp = 0; pp < peaks.size(); ++pp) {
        const auto peak = peaks[pp];

        // Compute the weighted average for localizing the peak, using a 5x5 window
        Vector2f peak_sum(0, 0);
        float weight_sum = 0.0f;
        for (int ii = -2; ii <= 2; ++ii) {
          for (int jj = -2; jj <= 2; ++jj) {
            const int row = peak[0] + ii;
            const int col = peak[1] + jj;

            if (col < 0 || col >= image.size[1] || row < 0 || row >= image.size[0]) {
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
            all_peaks.push_back(
              {peak[0] + kOffsetDueToUpsampling, peak[1] +
                kOffsetDueToUpsampling});
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
    for (size_t ii = 0; ii < channel_peaks[kNumVertexChannel - 1].size(); ++ii) {
      objects.push_back(DopeObject{channel_peaks[kNumVertexChannel - 1][ii]});
    }

    // Use 16 affinity field tensors (2 for each corner to centroid) to identify corner-centroid
    // associated for each corner peak
    for (size_t chan = 0; chan < kNumVertexChannel - 1; ++chan) {
      const std::vector<int> & peaks = channel_peaks[chan];
      for (size_t pp = 0; pp < peaks.size(); ++pp) {
        int best_idx = kInvalidId;
        float best_distance = kInvalidDist;
        float best_angle = kInvalidDist;

        for (size_t jj = 0; jj < objects.size(); ++jj) {
          const Vector2f & center = all_peaks[objects[jj].center];
          const Vector2f & point = all_peaks[peaks[pp]];
          const Vector2i point_int(static_cast<int>(point[0]), static_cast<int>(point[1]));

          Vector2f v_aff(
            maps[kNumVertexChannel + chan * 2].at<float>(point_int[1], point_int[0]),
            maps[kNumVertexChannel + chan * 2 + 1].at<float>(point_int[1], point_int[0]));
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
        image_coordinates.col(ii) = all_peaks[object.corners[valid_indices[ii]]] *
          kImageToMapScale;
      }
      image_coordinates.col(num_valid - 1) = all_peaks[object.center] * kImageToMapScale;
      output.push_back({std::move(valid_indices), std::move(image_coordinates)});
    }
    return output;
  }

  std::shared_ptr<geometry_msgs::msg::PoseArray> OnCallback(
    const isaac_ros_nvengine_interfaces::msg::TensorList::ConstSharedPtr & belief_maps_msg)
  {
    auto tensors = belief_maps_msg->tensors;
    std::vector<geometry_msgs::msg::Pose> poses;

    if (tensors.size() != kNumTensors) {
      RCLCPP_WARN(
        rclcpp::get_logger(logger_name_),
        "Error number of tensors: should have 1 tensor but got %d tensors", tensors.size());
      return FormatPoseArray(belief_maps_msg, poses);
    }

    auto tensor = tensors[0];
    // Check that the tensors are the right type
    for (size_t ii = 0; ii < tensors.size(); ++ii) {
      if (tensors[ii].data_type != 9) {  // see Tensor.msg
        RCLCPP_WARN(
          rclcpp::get_logger(
            logger_name_), "Error of type: should be 9 (float32) but is %d",
          tensors[ii].data_type);
        return FormatPoseArray(belief_maps_msg, poses);
      }
    }

    // Copy tensor data over to a more portable form
    std::array<cv::Mat, kInputMapsChannels> maps;
    for (size_t chan = 0; chan < kInputMapsChannels; ++chan) {
      maps[chan] = cv::Mat(kInputMapsRow, kInputMapsColumn, CV_32F);
      const size_t stride = kInputMapsColumn * kInputMapsRow * sizeof(float);
      std::memcpy(
        maps[chan].data, tensor.data.data() + chan * stride,
        stride);
    }

    // Analyze the belief map to find vertex locations in image space
    const std::vector<DopeObjectKeypoints> dope_objects = findObjects(maps);
    if (dope_objects.empty()) {
      RCLCPP_INFO(rclcpp::get_logger(logger_name_), "No objects detected.");
      return FormatPoseArray(belief_maps_msg, poses);
    }

    // Run Perspective-N-Point on the detected object to find the 6-DoF pose of the bounding cuboid
    const auto cuboid_3d_points = CuboidVertices(cuboid_dimensions_);
    for (const DopeObjectKeypoints & object : dope_objects) {
      const auto & valid_points = object.first;
      const size_t num_valid_points = valid_points.size();
      Eigen::Matrix3Xd keypoints_3d(3, num_valid_points);
      for (size_t j = 0; j < num_valid_points; ++j) {
        keypoints_3d.col(j) = cuboid_3d_points.col(valid_points[j]);
      }

      Pose3d pose;
      cv::Mat rvec, tvec;
      cv::Mat camera_matrix(3, 3, CV_64FC1);
      std::memcpy(
        camera_matrix.data, camera_matrix_.data(),
        camera_matrix_.size() * sizeof(double));
      cv::Mat dist_coeffs = cv::Mat::zeros(1, 4, CV_64FC1);  // no distortion

      cv::Mat cv_keypoints_3d;
      cv::eigen2cv(keypoints_3d, cv_keypoints_3d);
      cv::Mat cv_keypoints_2d;
      cv::eigen2cv(object.second, cv_keypoints_2d);
      if (!cv::solvePnP(
          cv_keypoints_3d.t(), cv_keypoints_2d.t(), camera_matrix,
          dist_coeffs, rvec, tvec))
      {
        RCLCPP_WARN(rclcpp::get_logger(logger_name_), "cv::solvePnP failed");
        return FormatPoseArray(belief_maps_msg, poses);
      }
      cv::cv2eigen(tvec, pose.translation);

      cv::Mat R;
      cv::Rodrigues(rvec, R);  // R is 3x3
      Eigen::Matrix3d e_mat;
      cv::cv2eigen(R, e_mat);
      pose.rotation = Eigen::Quaterniond(e_mat);

      // If the Z coordinate is negative, the pose is placing the object behind the camera (which
      // is incorrect), so we flip it
      if (pose.translation[2] < 0.f) {
        pose = pose.inverse();
      }

      constexpr double kCentimeterToMeter = 100.0;
      geometry_msgs::msg::Pose new_pose;
      new_pose.position.x = pose.translation[0] / kCentimeterToMeter;
      new_pose.position.y = pose.translation[1] / kCentimeterToMeter;
      new_pose.position.z = pose.translation[2] / kCentimeterToMeter;
      new_pose.orientation.x = pose.rotation.x();
      new_pose.orientation.y = pose.rotation.y();
      new_pose.orientation.z = pose.rotation.z();
      new_pose.orientation.w = pose.rotation.w();
      poses.push_back(new_pose);
    }

    return FormatPoseArray(belief_maps_msg, poses);
  }
};


DopeDecoderNode::DopeDecoderNode(rclcpp::NodeOptions options)
: Node("dope_decoder_node", options),
  // Parameters
  queue_size_(declare_parameter<int>("queue_size", rmw_qos_profile_default.depth)),
  header_frame_id_(declare_parameter<std::string>("frame_id", "")),
  config_filename_(declare_parameter<std::string>("configuration_file", "dope_config.yaml")),
  object_name_(declare_parameter<std::string>("object_name", "Ketchup")),
  // Subscribers
  belief_maps_sub_(create_subscription<isaac_ros_nvengine_interfaces::msg::TensorList>(
      "belief_map_array", queue_size_,
      std::bind(&DopeDecoderNode::DopeDecoderCallback, this, std::placeholders::_1))),
  // Publishers
  pub_(create_publisher<geometry_msgs::msg::PoseArray>("dope/pose_array", 1)),
  // Impl initialization
  impl_(std::make_unique<struct DopeDecoderImpl>(* this))
{
  if (header_frame_id_.empty()) {  // Received empty header frame id
    RCLCPP_WARN(get_logger(), "Received empty frame id! Header will be published without one.");
  }

  std::array<double, 3> cuboid_dimensions = {0};
  std::array<double, 9> camera_matrix = {0};

  // Open configuration YAML file
  const std::string package_directory = ament_index_cpp::get_package_share_directory(
    "isaac_ros_dope");
  fs::path yaml_path = package_directory / fs::path("config") / fs::path(config_filename_);
  if (!fs::exists(yaml_path)) {
    RCLCPP_ERROR(this->get_logger(), "%s could not be found. Exiting.", yaml_path.string().c_str());
  } else {
    // Parse parameters
    rcl_params_t * dope_params = rcl_yaml_node_struct_init(rcutils_get_default_allocator());
    rcl_parse_yaml_file(yaml_path.c_str(), dope_params);

    bool success = true;
    const std::string dimensions_param = "dimensions." + object_name_;
    rcl_variant_t * dimensions =
      rcl_yaml_node_struct_get("dope", dimensions_param.c_str(), dope_params);
    if (dimensions->double_array_value != nullptr) {
      for (int dd = 0; dd < 3; ++dd) {
        cuboid_dimensions[dd] = dimensions->double_array_value->values[dd];
      }
    } else {
      RCLCPP_ERROR(
        this->get_logger(), "No dimensions parameter found for object name: %s",
        object_name_.c_str());
      success = false;
    }

    if (success) {
      rcl_variant_t * cam_mat = rcl_yaml_node_struct_get("dope", "camera_matrix", dope_params);
      if (cam_mat->double_array_value != nullptr) {
        auto vv = cam_mat->double_array_value->values;
        camera_matrix = {vv[0], vv[1], vv[2], vv[3], vv[4], vv[5], vv[6], vv[7], vv[8]};
      } else {
        RCLCPP_ERROR(this->get_logger(), "No camera_matrix parameter found");
        success = false;
      }
    }
    rcl_yaml_node_struct_fini(dope_params);

    if (success) {
      impl_->Initialize(cuboid_dimensions, camera_matrix);
    } else {
      RCLCPP_ERROR(this->get_logger(), "Failed to initialize");
      throw std::runtime_error("Parameter parsing failure.");
    }
  }
}

DopeDecoderNode::~DopeDecoderNode() = default;

void DopeDecoderNode::DopeDecoderCallback(
  const isaac_ros_nvengine_interfaces::msg::TensorList::ConstSharedPtr belief_maps_msg)
{
  std::shared_ptr<geometry_msgs::msg::PoseArray> msg = impl_->OnCallback(belief_maps_msg);
  msg->header.frame_id = header_frame_id_;
  pub_->publish(*msg);
}

}  // namespace dope
}  // namespace isaac_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros::dope::DopeDecoderNode)
