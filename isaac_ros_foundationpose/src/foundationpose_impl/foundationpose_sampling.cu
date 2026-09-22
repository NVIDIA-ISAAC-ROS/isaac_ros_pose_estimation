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

#include "foundationpose_sampling.cu.hpp"

#include <cstdio>

namespace nvidia {
namespace isaac_ros {

__global__ void erode_depth_kernel(
    const float* depth, float* out, int H, int W, int depth_stride, int radius,
    float depth_diff_thres, float ratio_thres, float zfar) {
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  int w = blockIdx.x * blockDim.x + threadIdx.x;

  if (w >= W || h >= H) {
    return;
  }

  float d_ori = depth[h * depth_stride + w];

  // Check the validity of the depth value
  if (d_ori < 0.1f || d_ori >= zfar) {
    out[h * W + w] = 0.0f;
    return;
  }

  float bad_cnt = 0.0f;
  float total = 0.0f;

  // Loop over the neighboring pixels
  for (int u = w - radius; u <= w + radius; u++) {
    if (u < 0 || u >= W) {
      continue;
    }
    for (int v = h - radius; v <= h + radius; v++) {
      if (v < 0 || v >= H) {
        continue;
      }
      float cur_depth = depth[v * depth_stride + u];

      total += 1.0f;

      if (cur_depth < 0.1f || cur_depth >= zfar || fabsf(cur_depth - d_ori) > depth_diff_thres) {
        bad_cnt += 1.0f;
      }
    }
  }

  // Check the ratio of bad pixels
  if ((bad_cnt / total) > ratio_thres) {
    out[h * W + w] = 0.0f;
  } else {
    out[h * W + w] = d_ori;
  }
}

__global__ void bilateral_filter_depth_kernel(
    const float* depth, float* out, int H, int W, int depth_stride,
    float zfar, int radius, float sigmaD, float sigmaR) {
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  int w = blockIdx.x * blockDim.x + threadIdx.x;

  if (w >= W || h >= H) {
    return;
  }

  out[h * W + w] = 0.0f;

  // Compute the mean depth of the neighboring pixels
  float mean_depth = 0.0f;
  int num_valid = 0;
  for (int u = w - radius; u <= w + radius; u++) {
    if (u < 0 || u >= W) {
      continue;
    }
    for (int v = h - radius; v <= h + radius; v++) {
      if (v < 0 || v >= H) {
        continue;
      }
      // Get the current depth value
      float cur_depth = depth[v * depth_stride + u];
      if (cur_depth >= 0.1f && cur_depth < zfar) {
        num_valid++;
        mean_depth += cur_depth;
      }
    }
  }

  // Check if there are any valid pixels
  if (num_valid == 0) {
    return;
  }

  mean_depth /= (float)num_valid;

  float depthCenter = depth[h * depth_stride + w];
  float sum_weight = 0.0f;
  float sum = 0.0f;

  // Loop over the neighboring pixels again
  for (int u = w - radius; u <= w + radius; u++) {
    if (u < 0 || u >= W) {
      continue;
    }
    for (int v = h - radius; v <= h + radius; v++) {
      if (v < 0 || v >= H) {
        continue;
      }
      float cur_depth = depth[v * depth_stride + u];
      if (cur_depth >= 0.1f && cur_depth < zfar && fabsf(cur_depth - mean_depth) < 0.01f) {
        float weight = expf(
            -((float)((u - w) * (u - w) + (v - h) * (v - h))) / (2.0f * sigmaD * sigmaD) -
            (depthCenter - cur_depth) * (depthCenter - cur_depth) / (2.0f * sigmaR * sigmaR));
        sum_weight += weight;
        sum += weight * cur_depth;
      }
    }
  }

  // Check if the sum of weights is positive and the number of valid pixels is positive
  if (sum_weight > 0.0f && num_valid > 0) {
    out[h * W + w] = sum / sum_weight;
  }
}

uint16_t ceil_div(uint16_t numerator, uint16_t denominator) {
  uint32_t accumulator = numerator + denominator - 1;
  return accumulator / denominator;
}

void erode_depth(
    cudaStream_t stream, foundationpose::DeviceImageView<const float> depth,
    float* out, int radius, float depth_diff_thres, float ratio_thres, float zfar) {
  const int H = static_cast<int>(depth.size.height);
  const int W = static_cast<int>(depth.size.width);
  const int depth_stride = static_cast<int>(depth.row_stride_bytes / sizeof(float));
  dim3 block(16, 16);
  dim3 grid(ceil_div(W, 16), ceil_div(H, 16), 1);

  erode_depth_kernel<<<grid, block, 0, stream>>>(
      depth.data, out, H, W, depth_stride, radius, depth_diff_thres, ratio_thres, zfar);
}

void bilateral_filter_depth(
    cudaStream_t stream, foundationpose::DeviceImageView<const float> depth,
    float* out, float zfar, int radius, float sigmaD, float sigmaR) {
  const int H = static_cast<int>(depth.size.height);
  const int W = static_cast<int>(depth.size.width);
  const int depth_stride = static_cast<int>(depth.row_stride_bytes / sizeof(float));
  dim3 block(16, 16);
  dim3 grid(ceil_div(W, 16), ceil_div(H, 16), 1);

  bilateral_filter_depth_kernel<<<grid, block, 0, stream>>>(
      depth.data, out, H, W, depth_stride, zfar, radius, sigmaD, sigmaR);
}

__global__ void depth_to_xyz_map_kernel(
    const float* __restrict__ depth, float* __restrict__ xyz_map,
    int H, int W, int depth_stride, int xyz_stride,
    float fx, float fy, float cx, float cy) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  int v = blockIdx.y * blockDim.y + threadIdx.y;
  if (u >= W || v >= H) {
    return;
  }
  int pidx = v * depth_stride + u;
  int oidx = v * xyz_stride + u * 3;
  float z = depth[pidx];
  if (!(z > 0.0f)) {
    xyz_map[oidx + 0] = 0.0f;
    xyz_map[oidx + 1] = 0.0f;
    xyz_map[oidx + 2] = 0.0f;
    return;
  }
  float x = (static_cast<float>(u) - cx) * z / fx;
  float y = (static_cast<float>(v) - cy) * z / fy;
  xyz_map[oidx + 0] = x;
  xyz_map[oidx + 1] = y;
  xyz_map[oidx + 2] = z;
}

void depth_to_xyz_map(
    cudaStream_t stream, foundationpose::DeviceImageView<const float> depth,
    foundationpose::DeviceImageView<float> xyz_map,
    foundationpose::CameraIntrinsics intrinsics) {
  const int H = static_cast<int>(depth.size.height);
  const int W = static_cast<int>(depth.size.width);
  const int depth_stride = static_cast<int>(depth.row_stride_bytes / sizeof(float));
  const int xyz_stride = static_cast<int>(xyz_map.row_stride_bytes / sizeof(float));
  dim3 block(16, 16);
  dim3 grid(ceil_div(W, 16), ceil_div(H, 16), 1);
  depth_to_xyz_map_kernel<<<grid, block, 0, stream>>>(
      depth.data, xyz_map.data, H, W, depth_stride, xyz_stride,
      intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy);
}

// Block reduction: produces per-block min/max u, min/max v over mask>0 pixels and
// writes masked depths (where mask>0 AND depth>=min_depth) to a dense prefix of depth_scratch
// using a global atomic counter. A tiny kernel then resolves the global bbox and computes
// the final center (cx, cy, cz) on GPU. cz uses mean of valid depths (a GPU-friendly
// approximation of the CPU median that is numerically equivalent for centrally-clustered
// object segments and 2-3 orders of magnitude cheaper than a full sort).
__global__ void guess_translation_reduce_kernel(
    const float* __restrict__ depth, const uint8_t* __restrict__ mask,
    int H, int W, int depth_stride, int mask_stride, float min_depth,
    int* __restrict__ min_u, int* __restrict__ max_u,
    int* __restrict__ min_v, int* __restrict__ max_v,
    float* __restrict__ depth_sum, int* __restrict__ depth_count) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  int v = blockIdx.y * blockDim.y + threadIdx.y;
  if (u >= W || v >= H) {
    return;
  }
  if (mask[v * mask_stride + u] == 0) {
    return;
  }
  atomicMin(min_u, u);
  atomicMax(max_u, u);
  atomicMin(min_v, v);
  atomicMax(max_v, v);
  float d = depth[v * depth_stride + u];
  if (d >= min_depth) {
    atomicAdd(depth_sum, d);
    atomicAdd(depth_count, 1);
  }
}

__global__ void guess_translation_finalize_kernel(
    const int* min_u, const int* max_u, const int* min_v, const int* max_v,
    const float* depth_sum, const int* depth_count,
    float fx_inv_m00, float fx_inv_m02,
    float fy_inv_m11, float fy_inv_m12,
    float* center_and_flag) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  int mu = *min_u, Mu = *max_u, mv = *min_v, Mv = *max_v;
  int cnt = *depth_count;
  if (Mu < mu || Mv < mv || cnt <= 0) {
    center_and_flag[0] = 0.0f;
    center_and_flag[1] = 0.0f;
    center_and_flag[2] = 0.0f;
    center_and_flag[3] = 0.0f;
    return;
  }
  float uc = 0.5f * (static_cast<float>(mu) + static_cast<float>(Mu));
  float vc = 0.5f * (static_cast<float>(mv) + static_cast<float>(Mv));
  float zc = *depth_sum / static_cast<float>(cnt);
  center_and_flag[0] = (fx_inv_m00 * uc + fx_inv_m02) * zc;
  center_and_flag[1] = (fy_inv_m11 * vc + fy_inv_m12) * zc;
  center_and_flag[2] = zc;
  center_and_flag[3] = 1.0f;
}

__global__ void guess_translation_init_kernel(
    int* min_u, int* max_u, int* min_v, int* max_v,
    float* depth_sum, int* depth_count, int W, int H) {
  if (threadIdx.x != 0) {
    return;
  }
  *min_u = W;
  *max_u = -1;
  *min_v = H;
  *max_v = -1;
  *depth_sum = 0.0f;
  *depth_count = 0;
}

void guess_translation_gpu(
    cudaStream_t stream, foundationpose::DeviceImageView<const float> depth,
    foundationpose::DeviceImageView<const uint8_t> mask,
    foundationpose::CameraIntrinsics intrinsics,
    float min_depth,
    float* depth_scratch,
    float* center_and_flag_device) {
  const int H = static_cast<int>(depth.size.height);
  const int W = static_cast<int>(depth.size.width);
  const int depth_stride = static_cast<int>(depth.row_stride_bytes / sizeof(float));
  const int mask_stride = static_cast<int>(mask.row_stride_bytes);
  int* min_u = reinterpret_cast<int*>(depth_scratch);
  int* max_u = min_u + 1;
  int* min_v = min_u + 2;
  int* max_v = min_u + 3;
  float* depth_sum = reinterpret_cast<float*>(min_u + 4);
  int* depth_count = min_u + 5;

  guess_translation_init_kernel<<<1, 1, 0, stream>>>(min_u, max_u, min_v, max_v,
                                                     depth_sum, depth_count, W, H);

  dim3 block(16, 16);
  dim3 grid(ceil_div(W, 16), ceil_div(H, 16), 1);
  guess_translation_reduce_kernel<<<grid, block, 0, stream>>>(
      depth.data, mask.data, H, W, depth_stride, mask_stride, min_depth,
      min_u, max_u, min_v, max_v, depth_sum, depth_count);
  guess_translation_finalize_kernel<<<1, 1, 0, stream>>>(
      min_u, max_u, min_v, max_v, depth_sum, depth_count,
      1.0f / intrinsics.fx, -intrinsics.cx / intrinsics.fx,
      1.0f / intrinsics.fy, -intrinsics.cy / intrinsics.fy,
      center_and_flag_device);
}

__global__ void apply_deltas_kernel(
    float* __restrict__ poses, const float* __restrict__ trans_delta,
    const float* __restrict__ rot_delta, int num_poses,
    float half_diameter, float rot_normalizer) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_poses) {
    return;
  }
  float* P = poses + i * 16;

  float td0 = trans_delta[i * 3 + 0];
  float td1 = trans_delta[i * 3 + 1];
  float td2 = trans_delta[i * 3 + 2];
  td0 *= half_diameter; td1 *= half_diameter; td2 *= half_diameter;

  float r0 = tanhf(rot_delta[i * 3 + 0]) * rot_normalizer;
  float r1 = tanhf(rot_delta[i * 3 + 1]) * rot_normalizer;
  float r2 = tanhf(rot_delta[i * 3 + 2]) * rot_normalizer;
  float angle = sqrtf(r0 * r0 + r1 * r1 + r2 * r2);

  float Rm[9];
  if (angle > 1e-8f) {
    float inv = 1.0f / angle;
    float kx = r0 * inv, ky = r1 * inv, kz = r2 * inv;
    float s = sinf(angle), c = cosf(angle), C = 1.0f - c;
    // Standard Rodrigues rotation matrix (row-major) -- matches Eigen's AngleAxis.toRotationMatrix().
    float rm00 = c + kx * kx * C;
    float rm01 = kx * ky * C - kz * s;
    float rm02 = kx * kz * C + ky * s;
    float rm10 = ky * kx * C + kz * s;
    float rm11 = c + ky * ky * C;
    float rm12 = ky * kz * C - kx * s;
    float rm20 = kz * kx * C - ky * s;
    float rm21 = kz * ky * C + kx * s;
    float rm22 = c + kz * kz * C;
    // CPU version takes transpose() before multiplying pose. Transpose in-place:
    Rm[0] = rm00; Rm[1] = rm10; Rm[2] = rm20;
    Rm[3] = rm01; Rm[4] = rm11; Rm[5] = rm21;
    Rm[6] = rm02; Rm[7] = rm12; Rm[8] = rm22;
  } else {
    Rm[0] = 1.f; Rm[1] = 0.f; Rm[2] = 0.f;
    Rm[3] = 0.f; Rm[4] = 1.f; Rm[5] = 0.f;
    Rm[6] = 0.f; Rm[7] = 0.f; Rm[8] = 1.f;
  }

  // Eigen pose layout is column-major 4x4. Column k, row r -> index k*4 + r.
  // Rotation block is columns 0..2 rows 0..2. Translation is column 3 rows 0..2.
  float p00 = P[0], p10 = P[1], p20 = P[2];
  float p01 = P[4], p11 = P[5], p21 = P[6];
  float p02 = P[8], p12 = P[9], p22 = P[10];

  // new_rot = Rm * old_rot (both 3x3, row-major for Rm, column-major for P block).
  float n00 = Rm[0] * p00 + Rm[1] * p10 + Rm[2] * p20;
  float n10 = Rm[3] * p00 + Rm[4] * p10 + Rm[5] * p20;
  float n20 = Rm[6] * p00 + Rm[7] * p10 + Rm[8] * p20;
  float n01 = Rm[0] * p01 + Rm[1] * p11 + Rm[2] * p21;
  float n11 = Rm[3] * p01 + Rm[4] * p11 + Rm[5] * p21;
  float n21 = Rm[6] * p01 + Rm[7] * p11 + Rm[8] * p21;
  float n02 = Rm[0] * p02 + Rm[1] * p12 + Rm[2] * p22;
  float n12 = Rm[3] * p02 + Rm[4] * p12 + Rm[5] * p22;
  float n22 = Rm[6] * p02 + Rm[7] * p12 + Rm[8] * p22;

  P[0] = n00; P[1] = n10; P[2] = n20;
  P[4] = n01; P[5] = n11; P[6] = n21;
  P[8] = n02; P[9] = n12; P[10] = n22;

  P[12] += td0;
  P[13] += td1;
  P[14] += td2;
}

void apply_deltas_gpu(
    cudaStream_t stream, float* poses_device, const float* trans_delta, const float* rot_delta,
    int num_poses, float mesh_diameter, float rot_normalizer) {
  int threads = 128;
  int blocks = (num_poses + threads - 1) / threads;
  apply_deltas_kernel<<<blocks, threads, 0, stream>>>(
      poses_device, trans_delta, rot_delta, num_poses,
      0.5f * mesh_diameter, rot_normalizer);
}

}  // namespace isaac_ros
}  // namespace nvidia