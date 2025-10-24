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
#pragma once

#include <cstddef>
#include <set>

#include "extensions/centerpose/components/centerpose_detection.hpp"

namespace nvidia {
namespace isaac {
namespace centerpose {

enum class NMSMethod : int { ORIGINAL = 0, LINEAR = 1, GAUSSIAN = 2 };

std::set<size_t> SoftNMSNvidia(
    const float threshold, const float sigma, const float Nt, const NMSMethod method,
    CenterPoseDetectionList* src_detections);

}  // namespace centerpose
}  // namespace isaac
}  // namespace nvidia
