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

#include "foundationpose_sampling.hpp"

#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include <Eigen/Dense>
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
constexpr char kNamePoints[] = "points";
constexpr char RAW_CAMERA_MODEL_GXF_NAME[] = "intrinsics";
constexpr size_t kPoseMatrixLength = 4;
constexpr int kNumBatches = 6;
constexpr int kAngleStep = 30;

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

std::vector<Eigen::Matrix4f> GenerateSymmetricPoses(const std::vector<std::string>& symmetry_axes) {    
  std::vector<float> x_angles = {0.0};
  std::vector<float> y_angles = {0.0};
  std::vector<float> z_angles = {0.0};
  std::vector<Eigen::Matrix4f> symmetry_poses;

  if (symmetry_axes.empty()) {return symmetry_poses;}

  for (int i = 0; i < symmetry_axes.size(); ++i) {
    if (symmetry_axes[i].empty()) {
      continue;
    }

    std::string axis;
    std::string angle_str;
    std::vector<float> angle_degrees;

    // Parse format "axis_angle" (e.g., "x_30" or "x_full")
    size_t underscore_pos = symmetry_axes[i].find('_');
    if (underscore_pos == std::string::npos) {
      GXF_LOG_ERROR("[FoundationposeSampling] Invalid symmetry axis format %s. Expected 'axis_angle' (e.g., 'x_30' or 'x_full')", 
                    symmetry_axes[i].c_str());
      continue;
    }

    axis = symmetry_axes[i].substr(0, underscore_pos);
    angle_str = symmetry_axes[i].substr(underscore_pos + 1);

    if (angle_str == "full") {
      for (int angle = 0; angle < 360; angle += kAngleStep) {
        angle_degrees.push_back(angle / 180.0 * M_PI);
      }
    } else {
      try {
        float angle = std::stof(angle_str);
        angle_degrees.push_back(angle / 180.0 * M_PI);
      } catch (const std::exception& e) {
        GXF_LOG_ERROR("[FoundationposeSampling] Failed to parse angle from %s. Expected a number or 'full'", 
                      symmetry_axes[i].c_str());
        continue;
      }
    }
  
    if (axis == "x"){
      x_angles.insert(x_angles.end(), angle_degrees.begin(), angle_degrees.end());
    } else if (axis == "y") {
      y_angles.insert(y_angles.end(), angle_degrees.begin(), angle_degrees.end());
    } else if (axis == "z") {
      z_angles.insert(z_angles.end(), angle_degrees.begin(), angle_degrees.end());
    } else {
      GXF_LOG_ERROR("[FoundationposeSampling] Invalid symmetry axis %s. Expected x, y, or z", 
                    symmetry_axes[i].c_str());
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


// Filters poses to meet specified axis-angle constraints (format: "axis_angle" e.g., "x_30").
// Creates combinations of fixed angles while preserving original values for unconstrained axes.
std::vector<Eigen::Matrix4f> FilterPosesByConstraints(
    const std::vector<Eigen::Matrix4f>& poses_in, 
    const std::vector<std::string>& fixed_axis_angles) {
  
  if (fixed_axis_angles.empty()) {return poses_in;} // No constraints, return all poses

  // Parse constraints - collect all constraints for each axis
  std::vector<float> x_angles;
  std::vector<float> y_angles;
  std::vector<float> z_angles;

  for (const auto& angle_constraint : fixed_axis_angles) {
    if (angle_constraint.empty()) {
      continue;
    }

    std::string axis;
    float angle_degrees = 0.0f;
  
    // Parse format "axis_angle" (e.g., "x_30")
    size_t underscore_pos = angle_constraint.find('_');
    if (underscore_pos != std::string::npos) {
      axis = angle_constraint.substr(0, underscore_pos);
      std::string angle_str = angle_constraint.substr(underscore_pos + 1);
      try {
        angle_degrees = std::stof(angle_str);
      } catch (const std::exception& e) {
        GXF_LOG_ERROR("[FoundationposeSampling] Failed to parse angle from %s, skipping constraint", 
                     angle_constraint.c_str());
        continue;
      }
    } else {
      GXF_LOG_ERROR("[FoundationposeSampling] Invalid constraint format %s, expected axis_angle", 
                   angle_constraint.c_str());
      continue;
    }
  
    float angle_radians = angle_degrees * M_PI / 180.0f;
  
    if (axis == "x") {
      x_angles.push_back(angle_radians);
    } else if (axis == "y") {
      y_angles.push_back(angle_radians);
    } else if (axis == "z") {
      z_angles.push_back(angle_radians);
    } else {
      GXF_LOG_ERROR("[FoundationposeSampling] Invalid axis %s in constraint %s, expected x, y, or z", 
                   axis.c_str(), angle_constraint.c_str());
    }
  }

  // Check if any axis has constraints
  if (x_angles.empty() && y_angles.empty() && z_angles.empty()) {
    GXF_LOG_INFO("[FoundationposeSampling] No valid constraints provided");
    return poses_in;
  }

  // Create combinations of fixed angles (if any)
  std::vector<Eigen::Vector3f> fixed_angle_combinations;

  // Use NaN to represent unfixed angles
  float nan_value = std::numeric_limits<float>::quiet_NaN();

  // For each axis, either use the fixed values or a placeholder NaN
  std::vector<float> x_values = x_angles.empty() ? std::vector<float>{nan_value} : x_angles;
  std::vector<float> y_values = y_angles.empty() ? std::vector<float>{nan_value} : y_angles;
  std::vector<float> z_values = z_angles.empty() ? std::vector<float>{nan_value} : z_angles;

  // Generate all combinations of fixed angle values
  for (float x : x_values) {
    for (float y : y_values) {
      for (float z : z_values) {
        fixed_angle_combinations.push_back(Eigen::Vector3f(x, y, z));
        if (!std::isnan(x) || !std::isnan(y) || !std::isnan(z)) {
          GXF_LOG_DEBUG("[FoundationposeSampling] Fixed angle combination: X=%s Y=%s Z=%s", 
                      std::isnan(x) ? "unfixed" : std::to_string(x * 180.0 / M_PI).c_str(),
                      std::isnan(y) ? "unfixed" : std::to_string(y * 180.0 / M_PI).c_str(),
                      std::isnan(z) ? "unfixed" : std::to_string(z * 180.0 / M_PI).c_str());
        }
      }
    }
  }

  GXF_LOG_INFO("[FoundationposeSampling] Generated %lu fixed angle combinations", fixed_angle_combinations.size());

  // Map to store unique poses (indexed by rotation matrix as a string)
  std::map<std::string, Eigen::Matrix4f> unique_poses;

  // For each input pose and each fixed angle combination, create a new pose
  for (const auto& pose : poses_in) {
    // Extract original rotation and Euler angles
    Eigen::Matrix3f orig_rotation = pose.block<3, 3>(0, 0);
    Eigen::Vector3f orig_euler = orig_rotation.eulerAngles(0, 1, 2);

    for (const auto& fixed_angles : fixed_angle_combinations) {
      Eigen::Vector3f new_euler = orig_euler;
      if (!std::isnan(fixed_angles[0])) new_euler[0] = fixed_angles[0];
      if (!std::isnan(fixed_angles[1])) new_euler[1] = fixed_angles[1];
      if (!std::isnan(fixed_angles[2])) new_euler[2] = fixed_angles[2];

      Eigen::AngleAxisf rotX(new_euler[0], Eigen::Vector3f::UnitX());
      Eigen::AngleAxisf rotY(new_euler[1], Eigen::Vector3f::UnitY());
      Eigen::AngleAxisf rotZ(new_euler[2], Eigen::Vector3f::UnitZ());

      // Combine rotations (Z * Y * X order)
      Eigen::Matrix3f new_rotation = rotZ.toRotationMatrix() * 
                                     rotY.toRotationMatrix() * 
                                     rotX.toRotationMatrix();

      Eigen::Matrix4f new_pose = Eigen::Matrix4f::Identity();
      new_pose.block<3, 3>(0, 0) = new_rotation;
      new_pose.block<3, 1>(0, 3) = pose.block<3, 1>(0, 3);

      // Extract Euler angles from the new rotation matrix for the key
      Eigen::Vector3f key_angles = new_rotation.eulerAngles(0, 1, 2);

      int x_key = static_cast<int>(std::round(key_angles[0] * 180.0 / M_PI)) % 360;
      int y_key = static_cast<int>(std::round(key_angles[1] * 180.0 / M_PI)) % 360;
      int z_key = static_cast<int>(std::round(key_angles[2] * 180.0 / M_PI)) % 360;

      if (x_key < 0) x_key += 360;
      if (y_key < 0) y_key += 360;
      if (z_key < 0) z_key += 360;

      std::string angle_key = std::to_string(x_key) + "_" + 
                              std::to_string(y_key) + "_" + 
                              std::to_string(z_key);

      // Store the pose if not a duplicate
      if (unique_poses.find(angle_key) == unique_poses.end()) {
        unique_poses[angle_key] = new_pose;
      }
    }
  }

  // Convert unique poses map to output vector
  std::vector<Eigen::Matrix4f> poses_out;
  for (const auto& [key, pose] : unique_poses) {
    poses_out.push_back(pose);
  }
  // Log output size and example poses
  GXF_LOG_INFO("[FoundationposeSampling] Generated %lu unique poses after applying constraints", poses_out.size());

  return poses_out;
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

std::vector<Eigen::Matrix4f> MakeRotationGrid(const std::vector<std::string>& symmetry_axes, 
                                             const std::vector<std::string>& fixed_axis_angles, 
                                             unsigned int n_views = 40, 
                                             double inplane_step = 60.0) {
  auto cam_in_obs = SampleViewsIcosphere(n_views);
  GXF_LOG_DEBUG("[FoundationposeSampling] %lu poses generated from icosphere", cam_in_obs.size());

  inplane_step = inplane_step / 180.0 * M_PI;  // Convert degrees to radians
  std::vector<Eigen::Matrix4f> rot_grid;
  for (unsigned int i = 0; i < cam_in_obs.size(); i++) {
    for (double inplane_rot = 0; inplane_rot < 2.0 * M_PI; inplane_rot += inplane_step) {
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

  // Filter poses based on fixed axis angle constraints
  auto filtered_poses = FilterPosesByConstraints(rot_grid, fixed_axis_angles);
  GXF_LOG_DEBUG("[FoundationposeSampling] %lu poses left after applying fixed axis angle constraints", filtered_poses.size());

  std::vector<Eigen::Matrix4f> symmetry_tfs = GenerateSymmetricPoses(symmetry_axes);
  symmetry_tfs.push_back(Eigen::Matrix4f::Identity());
  auto clustered_poses = ClusterPoses(kAngleStep, 99999.0, filtered_poses, symmetry_tfs);
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
    symmetry_axes_, "symmetry_axes", "Symmetry Axes",
    "Rotational symmetries around axes with angles in format 'axis_angle' (e.g., 'x_30' for 30 degrees around x-axis). "
    "Valid axes are [x, y, z]. Angles must be explicitly specified (e.g., 'x_180') or use 'full' for full rotation.", 
    std::vector<std::string>{}, GXF_PARAMETER_FLAGS_DYNAMIC);

  result &= registrar->parameter(
    symmetry_planes_, "symmetry_planes", "Symmetry Planes (Deprecated)",
    "Deprecated: Use symmetry_axes instead. Adds 180-degree rotational symmetry around specified axes. "
    "Format: 'axis' (e.g., 'x' for 180-degree rotation around x-axis). Valid axes are [x, y, z].", 
    std::vector<std::string>{}, GXF_PARAMETER_FLAGS_DYNAMIC);

  result &= registrar->parameter(
    fixed_axis_angles_, "fixed_axis_angles", "Fixed Axis Angles",
    "Constraints on rotation angles in format 'axis_angle' (e.g., 'x_30' for 30 degrees around x-axis). "
    "Valid axes are [x, y, z]. Poses will be filtered to only include those matching these rotation constraints.", 
    std::vector<std::string>{}, GXF_PARAMETER_FLAGS_DYNAMIC);

  result &= registrar->parameter(
    fixed_translations_, "fixed_translations", "Fixed Translations",
    "Fixed translation components in format 'axis_value' (e.g., 'x_0.1', 'y_0.2', 'z_0.3'). "
    "Valid axes are [x, y, z] for translation in x, y, z directions respectively. "
    "If provided, these will override the automatically estimated translation components.",
    std::vector<std::string>{}, GXF_PARAMETER_FLAGS_DYNAMIC);

  result &= registrar->parameter(
      mesh_storage_, "mesh_storage", "Mesh Storage",
      "The mesh storage for mesh reuse");

  return gxf::ToResultCode(result);
}

gxf_result_t FoundationposeSampling::start() noexcept {
  GXF_LOG_DEBUG("[FoundationposeSampling] Start FoundationPose FoundationposeSampling");
  
  auto maybe_stream = cuda_stream_pool_.get()->allocateStream();
  if (!maybe_stream) { return gxf::ToResultCode(maybe_stream); }

  cuda_stream_handle_ = std::move(maybe_stream.value());
  if (!cuda_stream_handle_->stream()) {
    GXF_LOG_ERROR("[FoundationposeSampling] Allocated stream is not initialized!");
    return GXF_FAILURE;
  }
  if (!cuda_stream_handle_.is_null()) {
    cuda_stream_ = cuda_stream_handle_->stream().value();
  }

  return GXF_SUCCESS;
}

gxf_result_t FoundationposeSampling::tick() noexcept {
  GXF_LOG_DEBUG("[FoundationposeSampling] Tick FoundationPose FoundationposeSampling");

  // Combine symmetry_axes with converted symmetry_planes for backward compatibility
  std::vector<std::string> all_symmetry_axes = symmetry_axes_.get();
  
  // Convert symmetry_planes to symmetry_axes format (adding explicit 180-degree rotations)
  for (const auto& plane : symmetry_planes_.get()) {
    if (plane == "x" || plane == "y" || plane == "z") {
      all_symmetry_axes.push_back(plane + "_180");  // Explicitly specify 180 degrees
      GXF_LOG_WARNING("[FoundationposeSampling] Using deprecated symmetry_planes parameter. "
                      "Please use symmetry_axes instead (e.g., '%s_180').", plane.c_str());
    } else {
      GXF_LOG_ERROR("[FoundationposeSampling] Invalid symmetry plane axis %s, ignoring.", plane.c_str());
    }
  }

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

  // Try to reload mesh if the path has changed
  mesh_storage_.get()->TryReloadMesh();

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
  auto ob_in_cams = MakeRotationGrid(all_symmetry_axes, fixed_axis_angles_.get());
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

  erode_depth(cuda_stream_, reinterpret_cast<float*>(depth_handle->pointer()), erode_depth_device_, height, width);
  CHECK_CUDA_ERRORS(cudaGetLastError());

  bilateral_filter_depth(cuda_stream_, erode_depth_device_, bilateral_filter_depth_device_, height, width);
  CHECK_CUDA_ERRORS(cudaGetLastError());

  RowMajorMatrix8u mask;
  mask.resize(height, width);

  CHECK_CUDA_ERRORS(cudaMemcpyAsync(
    bilateral_filter_depth_host.data(), bilateral_filter_depth_device_,
    height * width * sizeof(float), cudaMemcpyDeviceToHost, cuda_stream_));

  CHECK_CUDA_ERRORS(cudaMemcpyAsync(
    mask.data(), segmentation_handle->pointer(), height * width * sizeof(uint8_t),
    cudaMemcpyDeviceToHost, cuda_stream_));
  cudaStreamSynchronize(cuda_stream_);

  Eigen::Vector3f center;
  
  // Always estimate translation first
  auto success = GuessTranslation(bilateral_filter_depth_host, mask, K, min_depth_, center);
  if (!success) {
    GXF_LOG_INFO("[FoundationposeSampling] Failed to guess translation. Not processing this image");
    return GXF_SUCCESS;
  }
  GXF_LOG_INFO("[FoundationposeSampling] Initial estimated translation: [x=%f, y=%f, z=%f]", 
               center[0], center[1], center[2]);

  // Override specific axes if fixed translations is provided
  auto fixed_translations = fixed_translations_.get();
  if (!fixed_translations.empty()) {
    // Parse fixed translation components
    for (const auto& component : fixed_translations) {
      if (component.empty()) {
        continue;
      }

      std::string axis;
      float value;
      
      // Parse format "axis_value" (e.g., "x_0.1")
      size_t underscore_pos = component.find('_');
      if (underscore_pos == std::string::npos) {
        GXF_LOG_ERROR("[FoundationposeSampling] Invalid fixed translations format '%s'. Expected 'axis_value' (e.g., 'x_0.1')", 
                      component.c_str());
        return GXF_FAILURE;
      }

      axis = component.substr(0, underscore_pos);
      try {
        value = std::stof(component.substr(underscore_pos + 1));
      } catch (const std::exception& e) {
        GXF_LOG_ERROR("[FoundationposeSampling] Failed to parse translation value from '%s'. Expected a number", 
                      component.c_str());
        return GXF_FAILURE;
      }

      // Override the appropriate component
      if (axis == "x") {
        center[0] = value;
      } else if (axis == "y") {
        center[1] = value;
      } else if (axis == "z") {
        center[2] = value;
      } else {
        GXF_LOG_ERROR("[FoundationposeSampling] Invalid translation axis '%s'. Expected x, y, or z", 
                      axis.c_str());
        return GXF_FAILURE;
      }
    }
    GXF_LOG_INFO("[FoundationposeSampling] Final translation after applying fixed components: [x=%f, y=%f, z=%f]", 
                 center[0], center[1], center[2]);
  }

  for (auto& m : ob_in_cams) {
    m.block<3, 1>(0, 3) = center;
  }

  // Add padding to the last batch to make the size divisible by kNumBatches
  auto remainder = ob_in_cams.size() % kNumBatches;
  auto padding_size = remainder == 0 ? 0 : kNumBatches - remainder;
  
  if (padding_size > 0) {
    GXF_LOG_INFO(
      "[FoundationposeSampling] Padding %d identity poses to make the size divisible by %d (current size: %lu)", 
      padding_size, kNumBatches, ob_in_cams.size());
    for (int i = 0; i < padding_size; i++) {
      ob_in_cams.push_back(Eigen::Matrix4f::Identity());
    }
  }

  // Flatten vector of eigen matrix into vector to make the memory continuous
  std::vector<float> ob_in_cams_vector;
  ob_in_cams_vector.reserve(ob_in_cams.size() * ob_in_cams[0].size());
  for (auto& mat : ob_in_cams) {
    std::vector<float> mat_data(mat.data(), mat.data() + mat.size());
    ob_in_cams_vector.insert(ob_in_cams_vector.end(), mat_data.begin(), mat_data.end());
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
        pose_arrays->pointer(), &ob_in_cams_vector[i*batch_size*16], pose_arrays->size(), cudaMemcpyHostToDevice, cuda_stream_));
    cudaStreamSynchronize(cuda_stream_);

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
