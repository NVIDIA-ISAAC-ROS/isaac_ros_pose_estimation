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

#include "foundationpose_utils.hpp"

#include <Eigen/Dense>
#include <vector>

#include "cuda.h"
#include "cuda_runtime.h"

namespace nvidia {
std::pair<Eigen::Vector3f, Eigen::Vector3f> FindMinMaxVertex(const aiMesh* mesh) {
  Eigen::Vector3f min_vertex = {0, 0, 0};
  Eigen::Vector3f max_vertex = {0, 0, 0};

  if (mesh->mNumVertices == 0) {
    return std::pair{min_vertex, max_vertex};
  }

  min_vertex << mesh->mVertices[0].x, mesh->mVertices[0].y, mesh->mVertices[0].z;
  max_vertex << mesh->mVertices[0].x, mesh->mVertices[0].y, mesh->mVertices[0].z;

  // Iterate over all vertices to find the bounding box
  for (size_t v = 0; v < mesh->mNumVertices; v++) {
    float vx = mesh->mVertices[v].x;
    float vy = mesh->mVertices[v].y;
    float vz = mesh->mVertices[v].z;

    min_vertex[0] = std::min(min_vertex[0], vx);
    min_vertex[1] = std::min(min_vertex[1], vy);
    min_vertex[2] = std::min(min_vertex[2], vz);

    max_vertex[0] = std::max(max_vertex[0], vx);
    max_vertex[1] = std::max(max_vertex[1], vy);
    max_vertex[2] = std::max(max_vertex[2], vz);
  }
  return std::pair{min_vertex, max_vertex};
}

float CalcMeshDiameter(const aiMesh* mesh) {
  float max_dist = 0.0;
  for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
    for (unsigned int j = i + 1; j < mesh->mNumVertices; ++j) {
      aiVector3D diff = mesh->mVertices[i] - mesh->mVertices[j];
      float dist = diff.Length();
      max_dist = std::max(max_dist, dist);
    }
  }
  return max_dist;
}

gxf::Expected<void> AddInputTimestampToOutput(gxf::Entity& output, gxf::Entity input) {
  std::string named_timestamp{"timestamp"};
  std::string unnamed_timestamp{""};
  auto maybe_input_timestamp = input.get<gxf::Timestamp>(named_timestamp.c_str());

  // Try to get a named timestamp from the input entity
  if (!maybe_input_timestamp) {
    maybe_input_timestamp = input.get<gxf::Timestamp>(unnamed_timestamp.c_str());
  }
  // If there is no named timestamp, try to get a unnamed timestamp from the input entity
  if (!maybe_input_timestamp) {
    GXF_LOG_ERROR("Failed to get input timestamp!");
    return gxf::ForwardError(maybe_input_timestamp);
  }

  // Try to get a named timestamp from the output entity
  auto maybe_output_timestamp = output.get<gxf::Timestamp>(named_timestamp.c_str());
  // If there is no named timestamp, try to get a unnamed timestamp from the output entity
  if (!maybe_output_timestamp) {
    maybe_output_timestamp = output.get<gxf::Timestamp>(unnamed_timestamp.c_str());
  }

  // If there is no unnamed timestamp also, then add a named timestamp to the output entity
  if (!maybe_output_timestamp) {
    maybe_output_timestamp = output.add<gxf::Timestamp>(named_timestamp.c_str());
    if (!maybe_output_timestamp) {
      GXF_LOG_ERROR("Failed to add timestamp to output message!");
      return gxf::ForwardError(maybe_output_timestamp);
    }
  }

  *maybe_output_timestamp.value() = *maybe_input_timestamp.value();
  return gxf::Success;
}

}  // namespace nvidia