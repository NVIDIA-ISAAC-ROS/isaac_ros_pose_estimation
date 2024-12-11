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

#include "foundationpose_sampling.hpp"
#include "foundationpose_utils.hpp"

#include <Eigen/Dense>

#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include "gxf/multimedia/camera.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace isaac_ros {

namespace {

constexpr char kNamePoses[] = "poses";
constexpr char kNamePoints[] = "points";
constexpr char RAW_CAMERA_MODEL_GXF_NAME[] = "intrinsics";
constexpr size_t kPoseMatrixLength = 4;
constexpr int kNumBatches = 6;

typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMajorMatrix8u;

// A helper function to create a vertex from a point
int AddVertex(const Eigen::Vector3f& p, std::vector<Eigen::Vector3f>& vertices) {
  vertices.push_back(p.normalized());
  return vertices.size() - 1;
}

// A helper function to create a face from three indices
void AddFace(int i, int j, int k, std::vector<Eigen::Vector3i>& faces) {
  faces.emplace_back(i, j, k);
}

// A helper function to get the middle point of two vertices
int GetMiddlePoint(
    int i, int j, std::vector<Eigen::Vector3f>& vertices, std::map<int64_t, int>& cache) {
  // check if the edge (i, j) has been processed before
  bool first_is_smaller = i < j;
  int64_t smaller = first_is_smaller ? i : j;
  int64_t greater = first_is_smaller ? j : i;
  int64_t key = (smaller << 32) + greater;

  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }

  // if not cached, create a new vertex
  Eigen::Vector3f p1 = vertices[i];
  Eigen::Vector3f p2 = vertices[j];
  Eigen::Vector3f pm = (p1 + p2) / 2.0;
  int index = AddVertex(pm, vertices);
  cache[key] = index;
  return index;
}

// A function to generate an icosphere
// Initial triangle values could found from https://sinestesia.co/blog/tutorials/python-icospheres/
std::vector<Eigen::Vector3f> GenerateIcosphere(unsigned int n_views) {
  std::map<int64_t, int> cache;
  std::vector<Eigen::Vector3f> vertices;
  std::vector<Eigen::Vector3i> faces;

  // create 12 vertices
  float t = (1.0 + std::sqrt(5.0)) / 2.0;  // the golden ratio
  AddVertex(Eigen::Vector3f(-1, t, 0), vertices);
  AddVertex(Eigen::Vector3f(1, t, 0), vertices);
  AddVertex(Eigen::Vector3f(-1, -t, 0), vertices);
  AddVertex(Eigen::Vector3f(1, -t, 0), vertices);
  AddVertex(Eigen::Vector3f(0, -1, t), vertices);
  AddVertex(Eigen::Vector3f(0, 1, t), vertices);
  AddVertex(Eigen::Vector3f(0, -1, -t), vertices);
  AddVertex(Eigen::Vector3f(0, 1, -t), vertices);
  AddVertex(Eigen::Vector3f(t, 0, -1), vertices);
  AddVertex(Eigen::Vector3f(t, 0, 1), vertices);
  AddVertex(Eigen::Vector3f(-t, 0, -1), vertices);
  AddVertex(Eigen::Vector3f(-t, 0, 1), vertices);

  // create 20 faces
  AddFace(0, 11, 5, faces);
  AddFace(0, 5, 1, faces);
  AddFace(0, 1, 7, faces);
  AddFace(0, 7, 10, faces);
  AddFace(0, 10, 11, faces);
  AddFace(1, 5, 9, faces);
  AddFace(5, 11, 4, faces);
  AddFace(11, 10, 2, faces);
  AddFace(10, 7, 6, faces);
  AddFace(7, 1, 8, faces);
  AddFace(3, 9, 4, faces);
  AddFace(3, 4, 2, faces);
  AddFace(3, 2, 6, faces);
  AddFace(3, 6, 8, faces);
  AddFace(3, 8, 9, faces);
  AddFace(4, 9, 5, faces);
  AddFace(2, 4, 11, faces);
  AddFace(6, 2, 10, faces);
  AddFace(8, 6, 7, faces);
  AddFace(9, 8, 1, faces);

  // subdivide each face into four smaller faces
  while (vertices.size() < n_views) {
    std::vector<Eigen::Vector3i> new_faces;
    for (const auto& face : faces) {
      int a = face[0];
      int b = face[1];
      int c = face[2];

      int ab = GetMiddlePoint(a, b, vertices, cache);
      int bc = GetMiddlePoint(b, c, vertices, cache);
      int ca = GetMiddlePoint(c, a, vertices, cache);

      AddFace(a, ab, ca, new_faces);
      AddFace(b, bc, ab, new_faces);
      AddFace(c, ca, bc, new_faces);
      AddFace(ab, bc, ca, new_faces);
    }
    faces = new_faces;
  }
  GXF_LOG_DEBUG("[FoundationposeSampling] Generated vertice size: %lu", vertices.size());
  return std::move(vertices);
}


float RotationGeodesticDistance(const Eigen::Matrix3f& R1, const Eigen::Matrix3f& R2) {
  float cos = ((R1 * R2.transpose()).trace() - 1) / 2.0;
  cos = std::max(std::min(cos, 1.0f), -1.0f);
  return std::acos(cos);
}

std::vector<Eigen::Matrix4f> GenerateSymmetricPoses(const std::vector<std::string>& symmetry_planes) {    
  float theta = 180.0 / 180.0 * M_PI;
  std::vector<float> x_angles = {0.0};
  std::vector<float> y_angles = {0.0};
  std::vector<float> z_angles = {0.0};
  std::vector<Eigen::Matrix4f> symmetry_poses;

  for (int i = 0; i < symmetry_planes.size(); ++i) {
    if (symmetry_planes[i] == "x"){
      x_angles.push_back(theta);
    } else if (symmetry_planes[i] == "y") {
      y_angles.push_back(theta);
    } else if (symmetry_planes[i] == "z") {
      z_angles.push_back(theta);
    } else {
    GXF_LOG_ERROR("[FoundationposeSampling] the input symmetry plane %s is invalid, ignore.", symmetry_planes[i]);
    continue;
    }
  }

  // Compute rotation matrix for each angle
  for (int i = 0; i < x_angles.size(); ++i) {
      auto rot_x = Eigen::AngleAxisf(x_angles[i], Eigen::Vector3f::UnitX());
    for (int j = 0; j < y_angles.size(); ++j) {
      auto rot_y = Eigen::AngleAxisf(y_angles[j], Eigen::Vector3f::UnitY());
      for (int k = 0; k < z_angles.size(); ++k) {
        auto rot_z = Eigen::AngleAxisf(z_angles[k], Eigen::Vector3f::UnitZ());
        auto rotaion_matrix_x = rot_x.toRotationMatrix();
        auto rotaion_matrix_y = rot_y.toRotationMatrix();
        auto rotaion_matrix_z = rot_z.toRotationMatrix();
        auto rotation = rotaion_matrix_z * rotaion_matrix_y * rotaion_matrix_x;
        Eigen::Matrix4f euler_matrix = Eigen::Matrix4f::Identity();
        euler_matrix.block<3, 3>(0, 0) = rotation;
        symmetry_poses.push_back(euler_matrix);
      }
    }
  }
  return std::move(symmetry_poses);
}

std::vector<Eigen::Matrix4f> ClusterPoses(
    float angle_diff, float dist_diff, std::vector<Eigen::Matrix4f>& poses_in,
    std::vector<Eigen::Matrix4f>& symmetry_tfs) {
  std::vector<Eigen::Matrix4f> poses_out;
  poses_out.push_back(poses_in[0]);
  const float radian_thres = angle_diff / 180.0 * M_PI;

  for (unsigned int i = 1; i < poses_in.size(); i++) {
    bool is_new = true;
    Eigen::Matrix4f cur_pose = poses_in[i];

    for (const auto& cluster : poses_out) {
      Eigen::Vector3f t0 = cluster.block(0, 3, 3, 1);
      Eigen::Vector3f t1 = cur_pose.block(0, 3, 3, 1);
      if ((t0 - t1).norm() >= dist_diff) {
        continue;
      }
      // Remove symmetry
      for (const auto& tf : symmetry_tfs) {
        Eigen::Matrix4f cur_pose_tmp = cur_pose * tf;
        float rot_diff =
            RotationGeodesticDistance(cur_pose_tmp.block(0, 0, 3, 3), cluster.block(0, 0, 3, 3));
        if (rot_diff < radian_thres) {
          is_new = false;
          break;
        }
      }
      if (!is_new) {
        break;
      }
    }

    if (is_new) {
      poses_out.push_back(poses_in[i]);
    }
  }
  return std::move(poses_out);
}

std::vector<Eigen::Matrix4f> SampleViewsIcosphere(unsigned int n_views) {
  auto vertices = GenerateIcosphere(n_views);
  std::vector<Eigen::Matrix4f, std::allocator<Eigen::Matrix4f>> cam_in_obs(
      vertices.size(), Eigen::Matrix4f::Identity(4, 4));
  for (unsigned int i = 0; i < vertices.size(); i++) {
    cam_in_obs[i].block<3, 1>(0, 3) = vertices[i];
    Eigen::Vector3f up(0, 0, 1);
    Eigen::Vector3f z_axis = -cam_in_obs[i].block<3, 1>(0, 3);
    z_axis.normalize();

    Eigen::Vector3f x_axis = up.cross(z_axis);
    if (x_axis.isZero()) {
      x_axis << 1, 0, 0;
    }
    x_axis.normalize();
    Eigen::Vector3f y_axis = z_axis.cross(x_axis);
    y_axis.normalize();
    cam_in_obs[i].block<3, 1>(0, 0) = x_axis;
    cam_in_obs[i].block<3, 1>(0, 1) = y_axis;
    cam_in_obs[i].block<3, 1>(0, 2) = z_axis;
  }
  return std::move(cam_in_obs);
}

std::vector<Eigen::Matrix4f> MakeRotationGrid(const std::vector<std::string>& symmetry_planes, unsigned int n_views = 40, int inplane_step = 60) {
  auto cam_in_obs = SampleViewsIcosphere(n_views);

  std::vector<Eigen::Matrix4f> rot_grid;
  for (unsigned int i = 0; i < cam_in_obs.size(); i++) {
    for (double inplane_rot = 0; inplane_rot < 360; inplane_rot += inplane_step) {
      Eigen::Matrix4f cam_in_ob = cam_in_obs[i];
      auto R_inplane = Eigen::Affine3f::Identity();
      R_inplane.rotate(Eigen::AngleAxisf(0, Eigen::Vector3f::UnitX()))
          .rotate(Eigen::AngleAxisf(0, Eigen::Vector3f::UnitY()))
          .rotate(Eigen::AngleAxisf(inplane_rot, Eigen::Vector3f::UnitZ()));

      cam_in_ob = cam_in_ob * R_inplane.matrix();
      Eigen::Matrix4f ob_in_cam = cam_in_ob.inverse();
      rot_grid.push_back(ob_in_cam);
    }
  }

  std::vector<Eigen::Matrix4f> symmetry_tfs = GenerateSymmetricPoses(symmetry_planes);
  symmetry_tfs.push_back(Eigen::Matrix4f::Identity());
  auto clustered_poses = ClusterPoses(30.0, 99999.0, rot_grid, symmetry_tfs);
  GXF_LOG_DEBUG("[FoundationposeSampling] %lu poses left after clustering", clustered_poses.size());
  return std::move(clustered_poses);
}

bool GuessTranslation(
    const Eigen::MatrixXf& depth, const RowMajorMatrix8u& mask, const Eigen::Matrix3f& K,
      float min_depth, Eigen::Vector3f& center) {
  // Find the indices where mask is positive
  std::vector<int> vs, us;
  for (int i = 0; i < mask.rows(); i++) {
    for (int j = 0; j < mask.cols(); j++) {
      if (mask(i, j) > 0) {
        vs.push_back(i);
        us.push_back(j);
      }
    }
  }
  if (us.empty()) {
    GXF_LOG_INFO("[FoundationposeSampling] Mask is all zero.");
    return false;
  }

  float uc =
      (*std::min_element(us.begin(), us.end()) + *std::max_element(us.begin(), us.end())) / 2.0;
  float vc =
      (*std::min_element(vs.begin(), vs.end()) + *std::max_element(vs.begin(), vs.end())) / 2.0;

  Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> valid =
      (mask.array() > 0) && (depth.array() >= min_depth);
  if (!valid.any()) {
    GXF_LOG_ERROR("[FoundationposeSampling] No valid value in mask.");
    return false;
  }

  std::vector<float> valid_depth;
  for (int i = 0; i < valid.rows(); i++) {
    for (int j = 0; j < valid.cols(); j++) {
      if (valid(i, j)) {
        valid_depth.push_back(depth(i, j));
      }
    }
  }
  std::sort(valid_depth.begin(), valid_depth.end());
  int n = valid_depth.size();
  float zc =
      (n % 2 == 0) ? (valid_depth[n / 2 - 1] + valid_depth[n / 2]) / 2.0 : valid_depth[n / 2];

  center = K.inverse() * Eigen::Vector3f(uc, vc, 1) * zc;
  return true;
}

}  // namespace

gxf_result_t FoundationposeSampling::registerInterface(gxf::Registrar* registrar) noexcept {
  gxf::Expected<void> result;

  result &= registrar->parameter(
      point_cloud_receiver_, "point_cloud_input", "Point Cloud Input",
      "The point cloud input as a tensor");

  result &= registrar->parameter(
      depth_receiver_, "depth_input", "Depth Image Input",
      "The depth image input as a video buffer");

  result &= registrar->parameter(
      segmentation_receiver_, "segmentation_input", "Segmentation Image Input",
      "The segmentation input as a video buffer");

  result &= registrar->parameter(
      rgb_receiver_, "rgb_input", "RGB image Input", "The RGB image input as a videobuffer");

  result &= registrar->parameter(
      posearray_transmitter_, "output", "PoseArray Output", "The ouput poses as a pose array");
    
  result &= registrar->parameter(
      camera_model_transmitter_, "camera_model_output", "Camera Model Output", "The camera model as a videobuffer");

  result &= registrar->parameter(
      point_cloud_transmitter_, "point_cloud_output", "Point Cloud Output",
      "The output point cloud as a tensor");

  result &= registrar->parameter(
      rgb_transmitter_, "rgb_output", "RGB Output",
      "The output RGB as a videobuffer");

  result &= registrar->parameter(
      allocator_, "allocator", "Allocator", "Output Allocator");

  result &= registrar->parameter(
      max_hypothesis_, "max_hypothesis", "Maximum number of hypothesis", 
      "Maximum number of pose hypothesis generated by sampling.");

  result &= registrar->parameter(
      cuda_stream_pool_, "cuda_stream_pool", "Cuda Stream Pool",
      "Instance of gxf::CudaStreamPool to allocate CUDA stream.");

  result &= registrar->parameter(
      min_depth_, "min_depth", "Minimum Depth",
      "Minimum depth value to consider for estimating object center in image space.",
      0.1f);
  
  result &= registrar->parameter(
    symmetry_planes_, "symmetry_planes", "Symmetry Planes",
    "Symmetry planes, select one or more from [x, y, z]. ", std::vector<std::string>{});

  return gxf::ToResultCode(result);
}

gxf_result_t FoundationposeSampling::start() noexcept {
  GXF_LOG_DEBUG("[FoundationposeSampling] Start FoundationPose FoundationposeSampling");
  
  auto maybe_stream = cuda_stream_pool_.get()->allocateStream();
  if (!maybe_stream) { return gxf::ToResultCode(maybe_stream); }

  stream_ = std::move(maybe_stream.value());
  if (!stream_->stream()) {
    GXF_LOG_ERROR("[FoundationposeSampling] allocated stream is not initialized!");
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

gxf_result_t FoundationposeSampling::tick() noexcept {
  GXF_LOG_DEBUG("[FoundationposeSampling] Tick FoundationPose FoundationposeSampling");

  auto maybe_xyz_message = point_cloud_receiver_->receive();
  if (!maybe_xyz_message) {
    GXF_LOG_ERROR("[FoundationposeSampling] Failed to receive point cloud message");
    return maybe_xyz_message.error();
  }

  auto maybe_depth_message = depth_receiver_->receive();
  if (!maybe_depth_message) {
    GXF_LOG_ERROR("[FoundationposeSampling] Failed to receive point cloud message");
    return maybe_depth_message.error();
  }

  auto maybe_segmentation_message = segmentation_receiver_->receive();
  if (!maybe_segmentation_message) {
    GXF_LOG_ERROR("[FoundationposeSampling] Failed to receive point cloud message");
    return maybe_segmentation_message.error();
  }

  auto maybe_rgb_message = rgb_receiver_->receive();
  if (!maybe_rgb_message) {
    GXF_LOG_ERROR("[FoundationposeSampling] Failed to receive point cloud message");
    return maybe_rgb_message.error();
  }

  auto maybe_depth_image = maybe_depth_message.value().get<gxf::VideoBuffer>();
  if (!maybe_depth_image) {
    GXF_LOG_ERROR("[FoundationposeSampling] Failed to get point cloud from message");
    return maybe_depth_image.error();
  }

  auto maybe_segmentation_image = maybe_segmentation_message.value().get<gxf::VideoBuffer>();
  if (!maybe_segmentation_image) {
    GXF_LOG_ERROR("[FoundationposeSampling] Failed to get point cloud from message");
    return maybe_segmentation_image.error();
  }

  cudaStream_t cuda_stream = 0;
  if (!stream_.is_null()) {
    cuda_stream = stream_->stream().value();
  }

  auto depth_handle = maybe_depth_image.value();
  auto depth_info = depth_handle->video_frame_info();

  auto segmentation_handle = maybe_segmentation_image.value();
  auto segmentation_info = segmentation_handle->video_frame_info();

  if (depth_info.width != segmentation_info.width ||
      depth_info.height != segmentation_info.height) {
    GXF_LOG_ERROR("[FoundationposeSampling] Input depth image and segmentation image have different dimension");
    return GXF_FAILURE;
  }

  if (depth_info.color_planes[0].stride != depth_info.width * sizeof(float)) {
    GXF_LOG_ERROR("[FoundationposeSampling] Expected no padding for depth image");
    return GXF_FAILURE;
  }

  if (segmentation_info.color_planes[0].stride != segmentation_info.width * sizeof(uint8_t)) {
    GXF_LOG_ERROR("[FoundationposeSampling] Expected no padding for segmenation image");
    return GXF_FAILURE;
  }

  // Get camera intrinsics from depth message
  auto maybe_gxf_camera_model = maybe_depth_message.value().get<nvidia::gxf::CameraModel>(
    RAW_CAMERA_MODEL_GXF_NAME);
  if (!maybe_gxf_camera_model) {
    GXF_LOG_ERROR("[FoundationposeSampling] Failed to receive image message");
    return maybe_gxf_camera_model.error();
  }
  auto gxf_camera_model = maybe_gxf_camera_model.value();
  Eigen::Matrix3f K;
  K << gxf_camera_model->focal_length.x, 0.0, gxf_camera_model->principal_point.x,
       0.0, gxf_camera_model->focal_length.y, gxf_camera_model->principal_point.y,
       0.0, 0.0, 1.0;

  const uint32_t height = depth_info.height;
  const uint32_t width = depth_info.width;

  if (!cached_) {
    int size = width * height * sizeof(float);
    CHECK_CUDA_ERRORS(cudaMalloc(&erode_depth_device_, size));
    CHECK_CUDA_ERRORS(cudaMalloc(&bilateral_filter_depth_device_, size));
    cached_ = true;
  }

  // Generate Pose Hypothesis
  auto ob_in_cams = MakeRotationGrid(symmetry_planes_.get());
  if (ob_in_cams.size() == 0 || ob_in_cams.size() > max_hypothesis_) {
    GXF_LOG_ERROR("[FoundationposeSampling] The size of rotation grid is not valid.");
    return GXF_FAILURE;
  }

  if (ob_in_cams[0].rows() != kPoseMatrixLength || ob_in_cams[0].cols() != kPoseMatrixLength) {
    GXF_LOG_ERROR("[FoundationposeSampling] The rotation grid dimension does not match pose matrix.");
    return GXF_FAILURE;
  }

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> bilateral_filter_depth_host;
  bilateral_filter_depth_host.resize(height, width);

  erode_depth(cuda_stream, reinterpret_cast<float*>(depth_handle->pointer()), erode_depth_device_, height, width);
  CHECK_CUDA_ERRORS(cudaGetLastError());

  bilateral_filter_depth(cuda_stream, erode_depth_device_, bilateral_filter_depth_device_, height, width);
  CHECK_CUDA_ERRORS(cudaGetLastError());

  RowMajorMatrix8u mask;
  mask.resize(height, width);

  CHECK_CUDA_ERRORS(cudaMemcpyAsync(
    bilateral_filter_depth_host.data(), bilateral_filter_depth_device_,
    height * width * sizeof(float), cudaMemcpyDeviceToHost, cuda_stream));

  CHECK_CUDA_ERRORS(cudaMemcpyAsync(
    mask.data(), segmentation_handle->pointer(), height * width * sizeof(uint8_t),
    cudaMemcpyDeviceToHost, cuda_stream));
  cudaStreamSynchronize(cuda_stream);

  Eigen::Vector3f center;
  auto success = GuessTranslation(bilateral_filter_depth_host, mask, K, min_depth_, center);
  if (!success) {
    GXF_LOG_INFO("[FoundationposeSampling] Failed to guess translation. Not processing this image");
    return GXF_SUCCESS;
  }
  for (auto& m : ob_in_cams) {
    m.block<3, 1>(0, 3) = center;
  }

  // Flatten vector of eigen matrix into vector to make the memory continuous
  std::vector<float> ob_in_cams_vector;
  ob_in_cams_vector.reserve(ob_in_cams.size() * ob_in_cams[0].size());
  for (auto& mat : ob_in_cams) {
    std::vector<float> mat_data(mat.data(), mat.data() + mat.size());
    ob_in_cams_vector.insert(ob_in_cams_vector.end(), mat_data.begin(), mat_data.end());
  }

  if (ob_in_cams.size() % kNumBatches !=0 ) {
    GXF_LOG_WARNING(
      "[FoundationposeSampling] The total pose size is not divisible by the iteration number, a few pose estimations might be dropped");
  }

  int batch_size = ob_in_cams.size() / kNumBatches;
  for (int i = 0; i < kNumBatches; i++) {
    // Allocate output message
    auto maybe_output_message = gxf::Entity::New(context());
    if (!maybe_output_message) {
      GXF_LOG_ERROR("[FoundationposeSampling] Failed to allocate PoseArray Message");
      return gxf::ToResultCode(maybe_output_message);
    }
    auto output_message = maybe_output_message.value();

    // Pose Array is column-major
    auto maybe_pose_arrays = output_message.add<gxf::Tensor>(kNamePoses);
    if (!maybe_pose_arrays) {
      GXF_LOG_ERROR("[FoundationposeSampling] Failed to add output Tensor");
      return gxf::ToResultCode(maybe_pose_arrays);
    }
    auto pose_arrays = maybe_pose_arrays.value();

    auto maybe_added_timestamp =
        AddInputTimestampToOutput(output_message, maybe_depth_message.value());
    if (!maybe_added_timestamp) {
      GXF_LOG_ERROR("[FoundationposeSampling] Failed to add timestamp");
      return gxf::ToResultCode(maybe_added_timestamp);
    }

    // Initialize output GXF tensor
    std::array<int32_t, nvidia::gxf::Shape::kMaxRank> pose_arrays_shape{
        static_cast<int32_t>(batch_size), kPoseMatrixLength, kPoseMatrixLength};
    auto result = pose_arrays->reshape<float>(
        nvidia::gxf::Shape{pose_arrays_shape, 3}, nvidia::gxf::MemoryStorageType::kDevice,
        allocator_);
    if (!result) {
      GXF_LOG_ERROR("[FoundationposeSampling] Failed to reshape pose array tensor");
      return gxf::ToResultCode(result);
    }

    auto expected_output_size = batch_size * kPoseMatrixLength * kPoseMatrixLength * sizeof(float);
    if (pose_arrays->size() != expected_output_size) {
      GXF_LOG_ERROR(
        "[FoundationposeSampling] Pose array output size %lu is not same as expected %lu", pose_arrays->size(), expected_output_size);
      return gxf::ToResultCode(result);
    }

    CHECK_CUDA_ERRORS(cudaMemcpyAsync(
        pose_arrays->pointer(), &ob_in_cams_vector[i*batch_size*16], pose_arrays->size(), cudaMemcpyHostToDevice, cuda_stream));
    cudaStreamSynchronize(cuda_stream);

    // Create entity, and forward camera model
    auto maybe_camera_model_out_message = gxf::Entity::New(context());
    if (!maybe_camera_model_out_message) {
      GXF_LOG_ERROR("[FoundationposeSampling] Unable to create entity");
      return gxf::ToResultCode(maybe_camera_model_out_message);
    }
    // Pose Array is column-major
    auto maybe_camera_model = maybe_camera_model_out_message.value().add<nvidia::gxf::CameraModel>(RAW_CAMERA_MODEL_GXF_NAME);
    if (!maybe_camera_model) {
      GXF_LOG_ERROR("[FoundationposeSampling] Failed to allocate output Tensor ");
      return gxf::ToResultCode(maybe_camera_model);
    }
    *maybe_camera_model.value() = *gxf_camera_model;
    posearray_transmitter_->publish(output_message);
    point_cloud_transmitter_->publish(maybe_xyz_message.value());
    rgb_transmitter_->publish(maybe_rgb_message.value());
    camera_model_transmitter_->publish(maybe_camera_model_out_message.value());
  }
  return GXF_SUCCESS;
}

gxf_result_t FoundationposeSampling::stop() noexcept { 
  CHECK_CUDA_ERRORS(cudaFree(erode_depth_device_));
  CHECK_CUDA_ERRORS(cudaFree(bilateral_filter_depth_device_));
  return GXF_SUCCESS;
}

}  // namespace isaac_ros
}  // namespace nvidia
