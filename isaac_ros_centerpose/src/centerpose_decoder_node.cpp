// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_centerpose/centerpose_decoder_node.hpp"

#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_centerpose/soft_nms_nvidia.hpp"
#include "isaac_ros_centerpose/cuboid_pnp_solver.hpp"
#include "isaac_ros_centerpose/cuboid3d.hpp"

#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/eigen.hpp"

#include "rclcpp/rclcpp.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace centerpose
{

namespace
{
constexpr char INPUT_TOPIC_NAME[] = "tensor_sub";
constexpr char CAMERA_INFO_INPUT_TOPIC_NAME[] = "camera_info";
constexpr char OUTPUT_TOPIC_NAME[] = "centerpose/detections";

constexpr std::array<const char *, 7> kTensorIdxToStr = {
  "bboxes", "scores", "kps", "clses", "obj_scale", "kps_displacement_mean", "kps_heatmap_mean"};

const std::unordered_map<const char *, int> kTensorStrToIdx = {
  {"bboxes", 0}, {"scores", 1}, {"kps", 2},
  {"clses", 3}, {"obj_scale", 4}, {"kps_displacement_mean", 5},
  {"kps_heatmap_mean", 6}};

constexpr float kNMSNt{0.5f};
constexpr float kNMSSigma{0.5f};
const NMSMethod kNMSMethod{NMSMethod::GAUSSIAN};

inline Eigen::MatrixXfRM PerformAffineTransform(
  const Eigen::Ref<const Eigen::MatrixXfRM> & untransformed_points,
  const Eigen::Matrix3fRM & affine_transform)
{
  Eigen::MatrixXfRM transformed_points =
    affine_transform.block<2, 2>(0, 0) * untransformed_points.transpose();
  transformed_points.colwise() += affine_transform.block<2, 1>(0, 2);
  return transformed_points.transpose();
}

inline Eigen::MatrixXfRM Calculate2DKeypoints(
  const Eigen::MatrixXfRM & kps_displacement_mean, const Eigen::Matrix3fRM & affine_transform)
{
  constexpr int32_t kps_displacement_rows{8};
  constexpr int32_t kps_displacement_cols{2};
  Eigen::Map<const Eigen::MatrixXfRM> reshaped_kps_displacement_mean{
    kps_displacement_mean.data(), kps_displacement_rows, kps_displacement_cols};
  return PerformAffineTransform(reshaped_kps_displacement_mean, affine_transform);
}

inline Eigen::MatrixXfRM CalculateBBoxPoints(
  const Eigen::MatrixXfRM & bbox, const Eigen::Matrix3fRM & affine_transform)
{
  constexpr int32_t bbox_rows{2};
  constexpr int32_t bbox_cols{2};
  Eigen::Map<const Eigen::MatrixXfRM> reshaped_bbox{bbox.data(), bbox_rows, bbox_cols};
  return PerformAffineTransform(reshaped_bbox, affine_transform);
}

std::unique_ptr<PnPResult> SolvePnP(
  const Eigen::MatrixXfRM & keypoints2d, const Eigen::MatrixXfRM & kps_heatmap_mean,
  const Cuboid3d & cuboid3d, const Eigen::Matrix3f & camera_matrix)
{
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
  auto result = pnp_solver.solvePnP(points_filtered);
  if (!result) {
    return nullptr;
  }
  return std::make_unique<PnPResult>(result.value());
}

Eigen::MatrixXfRM Calculate3DPoints(const PnPResult & pnp_result, const Cuboid3d & cuboid3d)
{
  Eigen::Matrix4f pose_pred = Eigen::Matrix4f::Identity();
  pose_pred.block<3, 3>(0, 0) = pnp_result.pose.orientation.normalized().toRotationMatrix();
  pose_pred.block<3, 1>(0, 3) = pnp_result.pose.position;

  const std::array<
    Eigen::Vector3f, static_cast<size_t>(CuboidVertexType::TOTAL_CORNER_VERTEX_COUNT)> &
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

Eigen::Matrix3fRM ComputeAffineTransform(
  const Eigen::Vector2f & eigen_center, const float scale_scalar, const float rot,
  const Eigen::Vector2i & output_field_size, bool inv = false)
{
  const cv::Point2f shift = {0.0f, 0.0f};

  const float src_w{scale_scalar};
  const float dst_w{static_cast<float>(output_field_size(0))};
  const float dst_h{static_cast<float>(output_field_size(1))};

  const float rot_rad{static_cast<float>(M_PI * rot / 180.0f)};
  auto calculate_direction = [](const cv::Point2f & pt, const float rot_rad) {
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

  auto calculate_third_point = [](const cv::Point2f & a, const cv::Point2f & b) {
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
  cv::Mat affine_matrix_cv = inv ? cv::getAffineTransform(dst_points, src_points) :
    cv::getAffineTransform(src_points, dst_points);
  Eigen::Matrix3dRM affine_matrix = Eigen::Matrix3dRM::Identity();
  cv::cv2eigen(affine_matrix_cv, affine_matrix);

  return affine_matrix.cast<float>();
}

}  // namespace

bool CenterPoseDecoderNode::UpdateCameraProperties(
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info)
{
  if (!camera_info) {
    RCLCPP_ERROR(get_logger(), "Received null camera info");
    return false;
  }

  camera_matrix_(0, 0) = camera_info->k[0];
  camera_matrix_(0, 2) = camera_info->k[2];

  camera_matrix_(1, 1) = camera_info->k[4];
  camera_matrix_(1, 2) = camera_info->k[5];

  camera_matrix_(2, 2) = 1.0f;

  // Avoid re-computing affine transform if it's not necessary
  if (original_image_size_.x() == static_cast<int32_t>(camera_info->width) &&
    original_image_size_.y() == static_cast<int32_t>(camera_info->height))
  {
    return true;
  }

  original_image_size_ = Eigen::Vector2i(camera_info->width, camera_info->height);

  const Eigen::Vector2f center = original_image_size_.cast<float>() / 2.0f;
  const float scale = std::max(
    static_cast<float>(camera_info->width),
    static_cast<float>(camera_info->height));

  constexpr float rotation_deg{0.0f};
  constexpr bool inverse{true};
  const Eigen::Vector2i output_field_size_eigen(
    static_cast<int>(output_field_size_[0]),
    static_cast<int>(output_field_size_[1]));
  affine_transform_ =
    ComputeAffineTransform(center, scale, rotation_deg, output_field_size_eigen, inverse);
  return true;
}

CenterPoseDecoderNode::CenterPoseDecoderNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("centerpose_decoder_node", options),
  output_field_size_{declare_parameter<std::vector<int64_t>>(
      "output_field_size",
      std::vector<int64_t>({}))},
  cuboid_scaling_factor_{declare_parameter<double>("cuboid_scaling_factor", 0.0)},
  score_threshold_{declare_parameter<double>("score_threshold", 1.0)},
  object_name_{declare_parameter<std::string>("object_name", "")},
  input_queue_size_(declare_parameter<int16_t>("input_queue_size", 10)),
  output_queue_size_(declare_parameter<int16_t>("output_queue_size", 10)),
  tensor_list_sub_{},
  camera_info_sub_{},
  camera_image_sync_{ExactPolicy{static_cast<uint32_t>(input_queue_size_)}, tensor_list_sub_,
    camera_info_sub_}
{
  if (output_field_size_.empty() || output_field_size_.size() != 2) {
    throw std::invalid_argument("Error: received invalid output field size");
  }

  if (cuboid_scaling_factor_ <= 0.0) {
    throw std::invalid_argument(
            "Error: received a less than or equal to zero cuboid scaling factor");
  }

  if (score_threshold_ >= 1.0) {
    throw std::invalid_argument(
            "Error: received score threshold greater or equal to 1.0");
  }

  if (object_name_.empty()) {
    RCLCPP_WARN(get_logger(), "Received empty object name. Defaulting to unknown.");
    object_name_ = "unknown";
  }

  // Create CUDA resources
  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("CenterPoseDecoderNode");

  // Set the QoS parameter for the publishers and subscribers.
  const rclcpp::QoS input_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "input_qos").keep_last(input_queue_size_);
  const rclcpp::QoS output_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "output_qos").keep_last(output_queue_size_);

  // Create subscribers and publishers
  rclcpp::SubscriptionOptions tensor_sub_options;
  tensor_sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  tensor_sub_options.acceptable_buffer_backends = "any";
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  camera_image_sync_.registerCallback(
    std::bind(
      &CenterPoseDecoderNode::InputCallback, this,
      std::placeholders::_1, std::placeholders::_2));

  tensor_list_sub_.subscribe(this, INPUT_TOPIC_NAME, input_qos, tensor_sub_options);
  camera_info_sub_.subscribe(this, CAMERA_INFO_INPUT_TOPIC_NAME, input_qos);

  detection3darray_pub_ = create_publisher<vision_msgs::msg::Detection3DArray>(
    OUTPUT_TOPIC_NAME, output_qos, pub_options);

  // internal initialization
  initialize();

  RCLCPP_DEBUG(get_logger(), "CenterPoseDecoderNode subscribers and publishers created");
}

bool CenterPoseDecoderNode::initialize()
{
  camera_matrix_ = Eigen::Matrix3f::Identity();
  original_image_size_ = Eigen::Vector2i{0, 0};
  return true;
}

CenterPoseDetectionList CenterPoseDecoderNode::ProcessTensor(
  const std::vector<Eigen::MatrixXfRM> & tensors)
{
  CenterPoseDetectionList detections;
  for (int i = 0; i < tensors[kTensorStrToIdx.at("scores")].rows(); ++i) {
    const float score{tensors[kTensorStrToIdx.at("scores")](i, 0)};
    const int cls{static_cast<int>(tensors[kTensorStrToIdx.at("clses")](i, 0))};
    if (score < score_threshold_) {
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
      cuboid_scaling_factor_ *
      tensors[kTensorStrToIdx.at("obj_scale")].block<1, obj_scale_size_flattened>(i, 0);
    detections.push_back(detection);
  }

  std::set<size_t> indices =
    SoftNMSNvidia(score_threshold_, kNMSSigma, kNMSNt, kNMSMethod, &detections);

  CenterPoseDetectionList filtered_detections;
  for (const size_t & idx : indices) {
    filtered_detections.push_back(detections[idx]);
  }

  for (CenterPoseDetection & detection : filtered_detections) {
    Cuboid3d cuboid3d{detection.bbox_size};
    auto maybe_pnp_result =
      SolvePnP(detection.keypoints2d, detection.kps_heatmap_mean, cuboid3d, camera_matrix_);
    if (!maybe_pnp_result) {
      continue;
    }
    PnPResult pnp_result = *maybe_pnp_result;
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

void CenterPoseDecoderNode::InputCallback(
  const TensorList::ConstSharedPtr & tensor_list,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info)
{
  RCLCPP_DEBUG(get_logger(), "CenterPoseDecoderNode input callback called");

  if (!UpdateCameraProperties(camera_info)) {
    return;
  }

  CenterPoseDetectionList detections;
  const cudaStream_t stream = *cuda_stream_;
  if (tensor_list->tensors.empty()) {
    RCLCPP_WARN(get_logger(), "Received empty tensor list");
    return;
  }
  if (tensor_list->names.size() != tensor_list->tensors.size()) {
    RCLCPP_ERROR(get_logger(), "Tensor names and tensors must have the same size");
    return;
  }

  std::array<const Tensor *, kTensorIdxToStr.size()> ordered_tensors{};
  for (size_t i = 0; i < kTensorIdxToStr.size(); ++i) {
    for (size_t j = 0; j < tensor_list->names.size(); ++j) {
      if (tensor_list->names[j] == kTensorIdxToStr[i]) {
        ordered_tensors[i] = &tensor_list->tensors[j];
        break;
      }
    }
    if (ordered_tensors[i] == nullptr) {
      RCLCPP_WARN(
        get_logger(), "Required tensor '%s' is missing from tensor list", kTensorIdxToStr[i]);
      return;
    }
  }

  const Tensor & tensor0 = *ordered_tensors[0];
  if (tensor0.shape.empty() || tensor0.shape[0] <= 0) {
    RCLCPP_WARN(get_logger(), "Tensor '%s' has empty shape", kTensorIdxToStr[0]);
    return;
  }
  const size_t batch_size = static_cast<size_t>(tensor0.shape[0]);

  for (size_t batch_i = 0; batch_i < batch_size; ++batch_i) {
    std::vector<Eigen::MatrixXfRM> batch_tensors;
    batch_tensors.reserve(ordered_tensors.size());

    for (const Tensor * tensor : ordered_tensors) {
      constexpr uint8_t kDLPackFloat = 2;
      if (tensor->dtype_code != kDLPackFloat || tensor->dtype_bits != 32 ||
        tensor->dtype_lanes != 1)
      {
        RCLCPP_ERROR(get_logger(), "CenterPose tensors must use float32 elements");
        return;
      }
      if (tensor->shape.size() != 3 ||
        tensor->shape[0] != static_cast<int64_t>(batch_size) ||
        tensor->shape[1] <= 0 || tensor->shape[2] <= 0)
      {
        RCLCPP_ERROR(get_logger(), "CenterPose tensors must have shape [batch, rows, columns]");
        return;
      }

      const Eigen::Index mat_rows = static_cast<Eigen::Index>(tensor->shape[1]);
      const Eigen::Index mat_cols = static_cast<Eigen::Index>(tensor->shape[2]);
      Eigen::MatrixXfRM mat(mat_rows, mat_cols);

      const size_t rows = static_cast<size_t>(mat_rows);
      const size_t columns = static_cast<size_t>(mat_cols);
      if (rows > std::numeric_limits<size_t>::max() / columns) {
        RCLCPP_ERROR(get_logger(), "CenterPose tensor size overflow");
        return;
      }
      const size_t elements_per_batch = rows * columns;
      if (elements_per_batch > std::numeric_limits<size_t>::max() / sizeof(float)) {
        RCLCPP_ERROR(get_logger(), "CenterPose tensor byte size overflow");
        return;
      }
      const size_t bytes_per_batch = elements_per_batch * sizeof(float);
      const size_t batch_stride = tensor->strides.empty() ?
        elements_per_batch : static_cast<size_t>(tensor->strides[0]);
      const bool invalid_inner_strides = !tensor->strides.empty() &&
        (tensor->strides.size() != tensor->shape.size() ||
        tensor->strides[0] < 0 || tensor->strides[1] != tensor->shape[2] ||
        tensor->strides[2] != 1);
      bool invalid_batch_stride = false;
      if (batch_stride < elements_per_batch) {
        invalid_batch_stride = true;
      }
      if (batch_stride > std::numeric_limits<size_t>::max() / sizeof(float)) {
        invalid_batch_stride = true;
      }
      if (invalid_inner_strides || invalid_batch_stride ||
        tensor->byte_offset > std::numeric_limits<size_t>::max())
      {
        RCLCPP_ERROR(get_logger(), "CenterPose tensors must have a contiguous inner layout");
        return;
      }
      const size_t batch_stride_bytes = batch_stride * sizeof(float);
      const size_t base_offset = static_cast<size_t>(tensor->byte_offset);
      if (batch_i > (std::numeric_limits<size_t>::max() - base_offset) / batch_stride_bytes) {
        RCLCPP_ERROR(get_logger(), "CenterPose tensor byte offset overflow");
        return;
      }
      const size_t byte_offset = base_offset + batch_i * batch_stride_bytes;
      if (byte_offset > tensor->data.size() ||
        bytes_per_batch > tensor->data.size() - byte_offset)
      {
        RCLCPP_ERROR(get_logger(), "CenterPose tensor data buffer is too small");
        return;
      }

      auto read_handle = cuda_buffer_backend::from_input_buffer(tensor->data, stream);
      cudaError_t cuda_result = cudaMemcpyAsync(
        mat.data(), read_handle.get_ptr() + byte_offset, bytes_per_batch,
        cudaMemcpyDefault, stream);
      if (cuda_result != cudaSuccess) {
        RCLCPP_ERROR(
          get_logger(), "Failed to copy tensor data: %s", cudaGetErrorString(cuda_result));
        return;
      }

      cuda_result = cudaStreamSynchronize(stream);
      if (cuda_result != cudaSuccess) {
        RCLCPP_ERROR(
          get_logger(), "Failed to synchronize CUDA stream: %s",
          cudaGetErrorString(cuda_result));
        return;
      }
      batch_tensors.push_back(std::move(mat));
    }

    CenterPoseDetectionList batch_detections = ProcessTensor(batch_tensors);
    detections.insert(
      detections.end(), batch_detections.begin(), batch_detections.end());
  }

  vision_msgs::msg::Detection3DArray detection3darray_message;
  detection3darray_message.header.stamp = camera_info->header.stamp;
  detection3darray_message.header.frame_id = camera_info->header.frame_id;
  detection3darray_message.detections.resize(detections.size());

  for (size_t i = 0; i < detections.size(); ++i) {
    vision_msgs::msg::Detection3D detection3d_message;
    detection3d_message.header.stamp = tensor_list->header.stamp;
    detection3d_message.header.frame_id = tensor_list->header.frame_id;
    if (detection3d_message.header.frame_id.empty()) {
      detection3d_message.header.frame_id = camera_info->header.frame_id;
    }
    detection3d_message.bbox.size.x = detections[i].bbox_size(0);
    detection3d_message.bbox.size.y = detections[i].bbox_size(1);
    detection3d_message.bbox.size.z = detections[i].bbox_size(2);
    detection3d_message.bbox.center.position.x = detections[i].position.x();
    detection3d_message.bbox.center.position.y = detections[i].position.y();
    detection3d_message.bbox.center.position.z = detections[i].position.z();
    detection3d_message.bbox.center.orientation.x = detections[i].quaternion.x();
    detection3d_message.bbox.center.orientation.y = detections[i].quaternion.y();
    detection3d_message.bbox.center.orientation.z = detections[i].quaternion.z();
    detection3d_message.bbox.center.orientation.w = detections[i].quaternion.w();

    vision_msgs::msg::ObjectHypothesisWithPose object_hypothesis_message;
    object_hypothesis_message.hypothesis.class_id =
      std::to_string(detections[i].class_id);
    object_hypothesis_message.hypothesis.score = detections[i].score;
    object_hypothesis_message.pose.pose = detection3d_message.bbox.center;
    detection3d_message.results.push_back(object_hypothesis_message);
    detection3darray_message.detections[i] = detection3d_message;
  }
  detection3darray_pub_->publish(detection3darray_message);
  RCLCPP_DEBUG(get_logger(), "CenterPoseDecoderNode detection3darray message published");
}


}  // namespace centerpose
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::centerpose::CenterPoseDecoderNode)
