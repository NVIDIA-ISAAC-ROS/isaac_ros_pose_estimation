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

#include "foundationpose_sampling.cu.hpp"

#include <cstdio>

namespace nvidia {
namespace isaac_ros {

__global__ void erode_depth_kernel(
    float* depth, float* out, int H, int W, int radius, float depth_diff_thres, float ratio_thres,
    float zfar) {
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  int w = blockIdx.x * blockDim.x + threadIdx.x;

  if (w >= W || h >= H) {
    return;
  }

  float d_ori = depth[h * W + w];

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
      float cur_depth = depth[v * W + u];

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
    float* depth, float* out, int H, int W, float zfar, int radius, float sigmaD, float sigmaR) {
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
      float cur_depth = depth[v * W + u];
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

  float depthCenter = depth[h * W + w];
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
      float cur_depth = depth[v * W + u];
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
    cudaStream_t stream, float* depth, float* out, int H, int W, int radius, float depth_diff_thres, float ratio_thres,
    float zfar) {
  dim3 block(16, 16);
  dim3 grid(ceil_div(W, 16), ceil_div(H, 16), 1);

  erode_depth_kernel<<<grid, block, 0, stream>>>(
      depth, out, H, W, radius, depth_diff_thres, ratio_thres, zfar);
}

void bilateral_filter_depth(
    cudaStream_t stream, float* depth, float* out, int H, int W, float zfar, int radius, float sigmaD, float sigmaR) {
  dim3 block(16, 16);
  dim3 grid(ceil_div(W, 16), ceil_div(H, 16), 1);

  bilateral_filter_depth_kernel<<<grid, block, 0, stream>>>(depth, out, H, W, zfar, radius, sigmaD, sigmaR);
}

}  // namespace isaac_ros
}  // namespace nvidia