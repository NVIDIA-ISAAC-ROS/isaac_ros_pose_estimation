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

#include "isaac_ros_foundationpose/foundationpose_impl/pose_sampler.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "Eigen/Dense"

#include "foundationpose_sampling.cu.hpp"
#include "isaac_ros_common/cuda_stream.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace foundationpose
{

namespace
{
constexpr int kAngleStep = 30;
constexpr size_t kPoseMatrixLength = 4;

int addVertex(const Eigen::Vector3f & p, std::vector<Eigen::Vector3f> & vertices)
{
  vertices.push_back(p.normalized());
  return static_cast<int>(vertices.size()) - 1;
}

int getMiddlePoint(
  int i, int j, std::vector<Eigen::Vector3f> & vertices, std::map<int64_t, int> & cache)
{
  bool first_smaller = i < j;
  int64_t smaller = first_smaller ? i : j;
  int64_t greater = first_smaller ? j : i;
  int64_t key = (smaller << 32) + greater;
  auto it = cache.find(key);
  if (it != cache.end()) {return it->second;}
  Eigen::Vector3f pm = (vertices[i] + vertices[j]) / 2.0f;
  int idx = addVertex(pm, vertices);
  cache[key] = idx;
  return idx;
}

std::vector<Eigen::Vector3f> generateIcosphere(unsigned int n_views)
{
  std::map<int64_t, int> cache;
  std::vector<Eigen::Vector3f> verts;
  std::vector<Eigen::Vector3i> faces;
  float t = (1.0f + std::sqrt(5.0f)) / 2.0f;
  auto av = [&](const Eigen::Vector3f & p) {addVertex(p, verts);};
  av({-1, t, 0}); av({1, t, 0}); av({-1, -t, 0}); av({1, -t, 0});
  av({0, -1, t}); av({0, 1, t}); av({0, -1, -t}); av({0, 1, -t});
  av({t, 0, -1}); av({t, 0, 1}); av({-t, 0, -1}); av({-t, 0, 1});

  auto af = [&](int a, int b, int c) {faces.emplace_back(a, b, c);};
  af(0, 11, 5); af(0, 5, 1); af(0, 1, 7); af(0, 7, 10); af(0, 10, 11);
  af(1, 5, 9); af(5, 11, 4); af(11, 10, 2); af(10, 7, 6); af(7, 1, 8);
  af(3, 9, 4); af(3, 4, 2); af(3, 2, 6); af(3, 6, 8); af(3, 8, 9);
  af(4, 9, 5); af(2, 4, 11); af(6, 2, 10); af(8, 6, 7); af(9, 8, 1);

  while (verts.size() < n_views) {
    std::vector<Eigen::Vector3i> new_faces;
    for (const auto & f : faces) {
      int ab = getMiddlePoint(f[0], f[1], verts, cache);
      int bc = getMiddlePoint(f[1], f[2], verts, cache);
      int ca = getMiddlePoint(f[2], f[0], verts, cache);
      new_faces.emplace_back(f[0], ab, ca);
      new_faces.emplace_back(f[1], bc, ab);
      new_faces.emplace_back(f[2], ca, bc);
      new_faces.emplace_back(ab, bc, ca);
    }
    faces = new_faces;
  }
  return verts;
}

float rotationGeodesicDistance(const Eigen::Matrix3f & R1, const Eigen::Matrix3f & R2)
{
  float c = ((R1 * R2.transpose()).trace() - 1.0f) / 2.0f;
  c = std::max(std::min(c, 1.0f), -1.0f);
  return std::acos(c);
}

std::vector<Eigen::Matrix4f> generateSymmetricPoses(const std::vector<std::string> & axes)
{
  std::vector<float> xa{0}, ya{0}, za{0};
  std::vector<Eigen::Matrix4f> result;
  if (axes.empty()) {return result;}
  for (const auto & s : axes) {
    if (s.empty()) {continue;}
    auto u = s.find('_');
    if (u == std::string::npos) {
      fprintf(stderr, "[PoseSampler] ERROR: Invalid symmetry_axes format '%s', "
        "expected 'axis_angle'\n", s.c_str());
      continue;
    }
    std::string axis = s.substr(0, u);
    std::string ang = s.substr(u + 1);
    std::vector<float> degs;
    if (ang == "full") {
      for (int a = 0; a < 360; a += kAngleStep) {
        degs.push_back(a * M_PI / 180.0f);
      }
    } else {
      try {
        degs.push_back(std::stof(ang) * M_PI / 180.0f);
      } catch (const std::exception &) {
        fprintf(stderr, "[PoseSampler] ERROR: Failed to parse angle from '%s'\n", s.c_str());
        continue;
      }
    }
    if (axis == "x") {
      xa.insert(xa.end(), degs.begin(), degs.end());
    } else if (axis == "y") {
      ya.insert(ya.end(), degs.begin(), degs.end());
    } else if (axis == "z") {
      za.insert(za.end(), degs.begin(), degs.end());
    } else {
      fprintf(stderr, "[PoseSampler] ERROR: Invalid symmetry axis '%s' in '%s'\n",
        axis.c_str(), s.c_str());
    }
  }
  for (float x : xa) {
    for (float y : ya) {
      for (float z : za) {
        Eigen::Matrix3f r = (Eigen::AngleAxisf(z, Eigen::Vector3f::UnitZ()) *
          Eigen::AngleAxisf(y, Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(x, Eigen::Vector3f::UnitX())).toRotationMatrix();
        Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
        m.block<3, 3>(0, 0) = r;
        result.push_back(m);
      }
    }
  }
  return result;
}

std::vector<Eigen::Matrix4f> clusterPoses(
  float angle_diff, std::vector<Eigen::Matrix4f> & poses,
  std::vector<Eigen::Matrix4f> & sym_tfs)
{
  std::vector<Eigen::Matrix4f> out;
  if (poses.empty()) {return out;}
  out.push_back(poses[0]);
  float thresh = angle_diff / 180.0f * M_PI;
  for (size_t i = 1; i < poses.size(); i++) {
    bool is_new = true;
    for (const auto & cl : out) {
      Eigen::Vector3f t0 = cl.block<3, 1>(0, 3);
      Eigen::Vector3f t1 = poses[i].block<3, 1>(0, 3);
      if ((t0 - t1).norm() >= 99999.0f) {continue;}
      for (const auto & tf : sym_tfs) {
        Eigen::Matrix4f tmp = poses[i] * tf;
        float rd = rotationGeodesicDistance(tmp.block<3, 3>(0, 0), cl.block<3, 3>(0, 0));
        if (rd < thresh) {is_new = false; break;}
      }
      if (!is_new) {break;}
    }
    if (is_new) {out.push_back(poses[i]);}
  }
  return out;
}

std::vector<Eigen::Matrix4f> filterPosesByConstraints(
  const std::vector<Eigen::Matrix4f> & poses_in,
  const std::vector<std::string> & fixed_axis_angles)
{
  if (fixed_axis_angles.empty()) {return poses_in;}

  std::vector<float> x_angles, y_angles, z_angles;
  for (const auto & constraint : fixed_axis_angles) {
    if (constraint.empty()) {continue;}
    auto u = constraint.find('_');
    if (u == std::string::npos) {continue;}
    std::string axis = constraint.substr(0, u);
    float angle_deg = 0.0f;
    try {
      angle_deg = std::stof(constraint.substr(u + 1));
    } catch (const std::exception &) {
      continue;
    }
    float angle_rad = angle_deg * static_cast<float>(M_PI) / 180.0f;
    if (axis == "x") {
      x_angles.push_back(angle_rad);
    } else if (axis == "y") {
      y_angles.push_back(angle_rad);
    } else if (axis == "z") {
      z_angles.push_back(angle_rad);
    }
  }
  if (x_angles.empty() && y_angles.empty() && z_angles.empty()) {return poses_in;}

  float nan_val = std::numeric_limits<float>::quiet_NaN();
  auto x_vals = x_angles.empty() ? std::vector<float>{nan_val} : x_angles;
  auto y_vals = y_angles.empty() ? std::vector<float>{nan_val} : y_angles;
  auto z_vals = z_angles.empty() ? std::vector<float>{nan_val} : z_angles;

  std::vector<Eigen::Vector3f> combos;
  for (float xv : x_vals) {
    for (float yv : y_vals) {
      for (float zv : z_vals) {
        combos.emplace_back(xv, yv, zv);
      }
    }
  }

  std::map<std::string, Eigen::Matrix4f> unique_poses;
  for (const auto & pose : poses_in) {
    Eigen::Vector3f orig_euler = pose.block<3, 3>(0, 0).eulerAngles(0, 1, 2);
    for (const auto & fixed : combos) {
      Eigen::Vector3f euler = orig_euler;
      if (!std::isnan(fixed[0])) {euler[0] = fixed[0];}
      if (!std::isnan(fixed[1])) {euler[1] = fixed[1];}
      if (!std::isnan(fixed[2])) {euler[2] = fixed[2];}

      Eigen::Matrix3f rot =
        (Eigen::AngleAxisf(euler[2], Eigen::Vector3f::UnitZ()) *
        Eigen::AngleAxisf(euler[1], Eigen::Vector3f::UnitY()) *
        Eigen::AngleAxisf(euler[0], Eigen::Vector3f::UnitX())).toRotationMatrix();

      Eigen::Matrix4f new_pose = Eigen::Matrix4f::Identity();
      new_pose.block<3, 3>(0, 0) = rot;
      new_pose.block<3, 1>(0, 3) = pose.block<3, 1>(0, 3);

      Eigen::Vector3f key_angles = rot.eulerAngles(0, 1, 2);
      int xk = static_cast<int>(std::round(key_angles[0] * 180.0 / M_PI)) % 360;
      int yk = static_cast<int>(std::round(key_angles[1] * 180.0 / M_PI)) % 360;
      int zk = static_cast<int>(std::round(key_angles[2] * 180.0 / M_PI)) % 360;
      if (xk < 0) {xk += 360;}
      if (yk < 0) {yk += 360;}
      if (zk < 0) {zk += 360;}
      std::string key = std::to_string(xk) + "_" + std::to_string(yk) + "_" + std::to_string(zk);
      if (unique_poses.find(key) == unique_poses.end()) {
        unique_poses[key] = new_pose;
      }
    }
  }

  std::vector<Eigen::Matrix4f> out;
  out.reserve(unique_poses.size());
  for (const auto & [key, pose] : unique_poses) {
    out.push_back(pose);
  }
  return out;
}

std::vector<Eigen::Matrix4f> makeRotationGrid(
  const std::vector<std::string> & symmetry_axes,
  const std::vector<std::string> & fixed_axis_angles,
  unsigned int n_views = 40, double inplane_step = 60.0)
{
  auto verts = generateIcosphere(n_views);
  std::vector<Eigen::Matrix4f> cam_in_obs(verts.size(), Eigen::Matrix4f::Identity());
  for (size_t i = 0; i < verts.size(); i++) {
    cam_in_obs[i].block<3, 1>(0, 3) = verts[i];
    Eigen::Vector3f up(0, 0, 1);
    Eigen::Vector3f z = -cam_in_obs[i].block<3, 1>(0, 3);
    z.normalize();
    Eigen::Vector3f x = up.cross(z);
    if (x.isZero()) {x << 1, 0, 0;}
    x.normalize();
    Eigen::Vector3f y = z.cross(x);
    y.normalize();
    cam_in_obs[i].block<3, 1>(0, 0) = x;
    cam_in_obs[i].block<3, 1>(0, 1) = y;
    cam_in_obs[i].block<3, 1>(0, 2) = z;
  }
  double step_rad = inplane_step / 180.0 * M_PI;
  std::vector<Eigen::Matrix4f> grid;
  for (const auto & c : cam_in_obs) {
    for (double r = 0; r < 2.0 * M_PI; r += step_rad) {
      Eigen::Matrix4f m = c;
      Eigen::Matrix4f rz = Eigen::Matrix4f::Identity();
      rz.block<3, 3>(0, 0) = Eigen::AngleAxisf(r, Eigen::Vector3f::UnitZ()).toRotationMatrix();
      m = m * rz;
      grid.push_back(m.inverse());
    }
  }
  auto filtered = filterPosesByConstraints(grid, fixed_axis_angles);

  auto sym_tfs = generateSymmetricPoses(symmetry_axes);
  sym_tfs.push_back(Eigen::Matrix4f::Identity());
  return clusterPoses(kAngleStep, filtered, sym_tfs);
}

}  // namespace

PoseSampler::PoseSampler(const PoseSamplerParams & params, cudaStream_t stream)
: params_(params), stream_(stream)
{
}

PoseSampler::~PoseSampler()
{
  if (erode_depth_device_) {cudaFree(erode_depth_device_);}
  if (bilateral_filter_depth_device_) {cudaFree(bilateral_filter_depth_device_);}
  if (center_flag_device_) {cudaFree(center_flag_device_);}
  if (center_flag_host_pinned_) {cudaFreeHost(center_flag_host_pinned_);}
}

SamplingResult PoseSampler::sample(
  const FrameObservationView & frame,
  std::shared_ptr<const MeshData> mesh_data)
{
  SamplingResult result;

  const ImageDim size = frame.depth.size;
  if (!device_mem_cached_ || !(cached_size_ == size)) {
    if (erode_depth_device_) {
      cudaFree(erode_depth_device_);
      erode_depth_device_ = nullptr;
    }
    if (bilateral_filter_depth_device_) {
      cudaFree(bilateral_filter_depth_device_);
      bilateral_filter_depth_device_ = nullptr;
    }
    size_t sz = size.numPixels() * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc(&erode_depth_device_, sz), "malloc erode");
    CHECK_CUDA_ERROR(cudaMalloc(&bilateral_filter_depth_device_, sz), "malloc bilateral");
    if (!center_flag_device_) {
      CHECK_CUDA_ERROR(cudaMalloc(&center_flag_device_, 4 * sizeof(float)), "malloc center");
      CHECK_CUDA_ERROR(cudaMallocHost(&center_flag_host_pinned_, 4 * sizeof(float)),
        "mallocHost center");
    }
    device_mem_cached_ = true;
    cached_size_ = size;
  }

  auto all_symmetry_axes = params_.symmetry_axes;
  for (const auto & plane : params_.symmetry_planes) {
    if (plane == "x" || plane == "y" || plane == "z") {
      all_symmetry_axes.push_back(plane + "_180");
      fprintf(stderr, "[PoseSampler] WARNING: symmetry_planes is deprecated. "
        "Use symmetry_axes '%s_180' instead.\n", plane.c_str());
    } else {
      fprintf(stderr, "[PoseSampler] ERROR: Invalid symmetry plane axis '%s', ignoring.\n",
        plane.c_str());
    }
  }
  auto ob_in_cams = makeRotationGrid(all_symmetry_axes, params_.fixed_axis_angles);
  if (ob_in_cams.empty() || ob_in_cams.size() > params_.max_hypothesis) {
    return result;
  }

  nvidia::isaac_ros::erode_depth(
    stream_, frame.depth, erode_depth_device_);
  CHECK_CUDA_ERROR(cudaGetLastError(), "erode_depth");

  DeviceImageView<const float> eroded_depth{
    erode_depth_device_, size,
    static_cast<size_t>(size.width) * sizeof(float), 1};
  nvidia::isaac_ros::bilateral_filter_depth(
    stream_, eroded_depth, bilateral_filter_depth_device_);
  CHECK_CUDA_ERROR(cudaGetLastError(), "bilateral_filter");

  DeviceImageView<const float> filtered_depth{
    bilateral_filter_depth_device_, size,
    static_cast<size_t>(size.width) * sizeof(float), 1};
  nvidia::isaac_ros::guess_translation_gpu(
    stream_, filtered_depth, frame.mask, frame.intrinsics, params_.min_depth,
    erode_depth_device_, center_flag_device_);
  CHECK_CUDA_ERROR(cudaGetLastError(), "guess_translation_gpu");

  CHECK_CUDA_ERROR(cudaMemcpyAsync(center_flag_host_pinned_, center_flag_device_,
      4 * sizeof(float), cudaMemcpyDeviceToHost, stream_), "d2h center");
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_), "sync center");
  if (center_flag_host_pinned_[3] <= 0.0f) {
    return result;
  }
  Eigen::Vector3f center(center_flag_host_pinned_[0], center_flag_host_pinned_[1],
    center_flag_host_pinned_[2]);

  // Apply fixed translations if provided
  for (const auto & comp : params_.fixed_translations) {
    if (comp.empty()) {continue;}
    auto u = comp.find('_');
    if (u == std::string::npos) {
      fprintf(stderr, "[PoseSampler] ERROR: Invalid fixed_translation format '%s', "
        "expected 'axis_value'\n", comp.c_str());
      continue;
    }
    std::string axis = comp.substr(0, u);
    try {
      float val = std::stof(comp.substr(u + 1));
      if (axis == "x") {
        center[0] = val;
      } else if (axis == "y") {
        center[1] = val;
      } else if (axis == "z") {
        center[2] = val;
      } else {
        fprintf(stderr, "[PoseSampler] ERROR: Invalid fixed_translation axis '%s'\n",
          axis.c_str());
      }
    } catch (const std::exception &) {
      fprintf(stderr, "[PoseSampler] ERROR: Failed to parse value from '%s'\n", comp.c_str());
      continue;
    }
  }

  for (auto & m : ob_in_cams) {
    m.block<3, 1>(0, 3) = center;
  }

  auto remainder = ob_in_cams.size() % kNumBatches;
  auto padding = remainder == 0 ? 0 : kNumBatches - remainder;
  for (size_t i = 0; i < padding; i++) {
    ob_in_cams.push_back(Eigen::Matrix4f::Identity());
  }

  std::vector<float> flat;
  flat.reserve(ob_in_cams.size() * 16);
  for (auto & m : ob_in_cams) {
    flat.insert(flat.end(), m.data(), m.data() + 16);
  }

  result.poses = std::move(flat);
  result.total_poses = static_cast<int32_t>(ob_in_cams.size());
  result.batch_size = result.total_poses / kNumBatches;
  result.num_batches = kNumBatches;
  return result;
}

}  // namespace foundationpose
}  // namespace isaac_ros
}  // namespace nvidia
