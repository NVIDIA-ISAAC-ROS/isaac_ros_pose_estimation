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

#include "foundationpose_decoder.cu.hpp"

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

namespace nvidia {
namespace isaac_ros {

// This function will find the index with the maximum score
int getMaxScoreIndex(cudaStream_t cuda_stream, float* scores, int N) {
  // Wrap raw pointers with device pointers
  thrust::device_ptr<float> dev_scores(scores);
  // Find the maximum score
  thrust::device_ptr<float> max_score_ptr =
      thrust::max_element(thrust::cuda::par.on(cuda_stream), dev_scores, dev_scores + N);

  // Calculate the index of the maximum score
  int max_index = max_score_ptr - dev_scores;
  return max_index;
}

}  // namespace isaac_ros
}  // namespace nvidia