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
#include "extensions/centerpose/components/soft_nms_nvidia.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <set>
#include <utility>
#include <vector>

#include "extensions/centerpose/components/centerpose_detection.hpp"

namespace nvidia {
namespace isaac {
namespace centerpose {

namespace {

size_t SearchForMaxScoreIndex(const CenterPoseDetectionList* detections, const size_t start_idx) {
  size_t max_idx{start_idx};
  float max_score{(*detections)[max_idx].score};
  for (size_t i = start_idx + 1; i < detections->size(); ++i) {
    if (max_score < (*detections)[i].score) {
      max_score = (*detections)[i].score;
      max_idx = i;
    }
  }
  return max_idx;
}

float ComputeNMSWeight(float ov, float sigma, float Nt, NMSMethod method) {
  switch (method) {
    case NMSMethod::LINEAR:
      return (ov > Nt) ? 1.0f - ov : 1.0f;
    case NMSMethod::GAUSSIAN:
      return std::exp(-(ov * ov) / sigma);
    case NMSMethod::ORIGINAL:
      return (ov > Nt) ? 0.0f : 1.0f;
  }
  return 1.0f;
}

std::vector<size_t> DoNMSIterations(
    const size_t start_idx, const float threshold, const float sigma, const float Nt,
    const NMSMethod method, size_t* total_detections, CenterPoseDetectionList* detections) {
  float tx1 = (*detections)[start_idx].bbox(0, 0);
  float ty1 = (*detections)[start_idx].bbox(0, 1);
  float tx2 = (*detections)[start_idx].bbox(1, 0);
  float ty2 = (*detections)[start_idx].bbox(1, 1);

  size_t curr_idx{start_idx + 1};

  std::vector<size_t> remove_indices;
  while (curr_idx < *total_detections) {
    float x1 = (*detections)[curr_idx].bbox(0, 0);
    float y1 = (*detections)[curr_idx].bbox(0, 1);
    float x2 = (*detections)[curr_idx].bbox(1, 0);
    float y2 = (*detections)[curr_idx].bbox(1, 1);
    float area = (x2 - x1 + 1) * (y2 - y1 + 1);
    float iw = std::min(tx2, x2) - std::max(tx1, x1) + 1;
    if (iw < 0) {
      continue;
    }
    float ih = std::min(ty2, y2) - std::max(ty1, y1) + 1;
    if (ih < 0) {
      continue;
    }
    float ua = static_cast<float>((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih);
    float ov = iw * ih / ua;  // iou between max box and detection box
    float weight = ComputeNMSWeight(ov, sigma, Nt, method);
    (*detections)[curr_idx].score = weight * (*detections)[curr_idx].score;
    if ((*detections)[curr_idx].score < threshold) {
      remove_indices.push_back(curr_idx);
      *total_detections = *total_detections - 1;
      curr_idx--;
    }
    curr_idx++;
  }
  return remove_indices;
}

}  // namespace

std::set<size_t> SoftNMSNvidia(
    const float threshold, const float sigma, const float Nt, const NMSMethod method,
    CenterPoseDetectionList* src_detections) {
  if (src_detections->empty()) {
    return {};
  }

  if (src_detections->size() == 1) {
    return {0};
  }

  size_t N = src_detections->size();
  std::set<size_t> total_remove_indices;

  for (size_t i = 0; i < N; ++i) {
    size_t max_idx{SearchForMaxScoreIndex(src_detections, i)};
    std::swap((*src_detections)[i], (*src_detections)[max_idx]);
    std::vector<size_t> remove_indices =
        DoNMSIterations(i, threshold, sigma, Nt, method, &N, src_detections);
    for (const size_t idx : remove_indices) {
      total_remove_indices.insert(total_remove_indices.end(), idx);
    }
  }

  std::set<size_t> all_indices;
  for (size_t i = 0; i < src_detections->size(); ++i) {
    all_indices.insert(all_indices.end(), i);
  }

  std::set<size_t> keep_indices;
  std::set_difference(
      all_indices.begin(), all_indices.end(), total_remove_indices.begin(),
      total_remove_indices.end(), std::inserter(keep_indices, keep_indices.end()));
  return keep_indices;
}

}  // namespace centerpose

}  // namespace isaac

}  // namespace nvidia
