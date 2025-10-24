// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mesh_storage.hpp"

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "foundationpose_utils.hpp"

namespace {
constexpr int kFixTextureMapColor = 128;
}  // namespace

namespace nvidia {
namespace isaac_ros {

gxf_result_t MeshStorage::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;

  result &= registrar->parameter(
      mesh_file_path_, "mesh_file_path", "Mesh File Path",
      "Path to the mesh file",
      gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_DYNAMIC);

  result &= registrar->parameter(
      cuda_stream_pool_, "cuda_stream_pool", "Cuda Stream Pool",
      "Instance of gxf::CudaStreamPool to allocate CUDA stream.");

  return gxf::ToResultCode(result);
}

gxf_result_t MeshStorage::initialize() {
  auto maybe_stream = cuda_stream_pool_.get()->allocateStream();
  if (!maybe_stream) { return gxf::ToResultCode(maybe_stream); }

  cuda_stream_handle_ = std::move(maybe_stream.value());
  if (!cuda_stream_handle_->stream()) {
    GXF_LOG_ERROR("[MeshStorage] Allocated stream is not initialized!");
    return GXF_FAILURE;
  }
  if (!cuda_stream_handle_.is_null()) {
    cuda_stream_ = cuda_stream_handle_->stream().value();
  }

  // Initialize mesh data
  mesh_data_ = std::make_shared<MeshData>();
  auto result = LoadMeshData(mesh_file_path_.get());
  if (result != GXF_SUCCESS) {
    GXF_LOG_ERROR("[MeshStorage] Failed to load mesh file");
    return result;
  }

  GXF_LOG_DEBUG("[MeshStorage] Successfully loaded mesh with %d vertices", 
                mesh_data_->num_vertices);
  GXF_LOG_DEBUG("[MeshStorage] Successfully loaded texture with %d x %d", 
                mesh_data_->texture_map_width, mesh_data_->texture_map_height);
  return GXF_SUCCESS;
}

gxf_result_t MeshStorage::LoadTextureData(const std::string& texture_file_path) {
  cv::Mat rgb_texture_map;

  if (!std::filesystem::exists(texture_file_path)) {
    if (mesh_data_->texture_path.empty() && mesh_data_->texture_map_device != nullptr && mesh_data_->texture_map_width == mesh_data_->num_vertices) {
      GXF_LOG_WARNING("[MeshStorage] %s could not be found, reuse the pure color texture map", texture_file_path.c_str());
      return GXF_SUCCESS;
    }
    GXF_LOG_WARNING("[MeshStorage] %s could not be found, assign texture map with pure color", texture_file_path.c_str());
    // The pure color texture map is actually a list of vertex colors
    rgb_texture_map = cv::Mat(1, mesh_data_->num_vertices, CV_8UC3,
                              cv::Scalar(kFixTextureMapColor, kFixTextureMapColor, kFixTextureMapColor));
    mesh_data_->has_tex = false;
  } else {
    mesh_data_->texture_path = texture_file_path;
    cv::Mat texture_map = cv::imread(texture_file_path);
    cv::cvtColor(texture_map, rgb_texture_map, cv::COLOR_BGR2RGB);
  }

  if (!rgb_texture_map.isContinuous()) {
    GXF_LOG_ERROR("[MeshStorage] Texture map is not continuous");
    return GXF_FAILURE;
  }

  // Free the previous texture map if it exists
  if (mesh_data_->texture_map_device != nullptr) {
    mesh_data_->FreeTextureDeviceMemory();
  }

  mesh_data_->texture_map_height = rgb_texture_map.rows;
  mesh_data_->texture_map_width = rgb_texture_map.cols;
  mesh_data_->texture_map_channels = rgb_texture_map.channels();

  CHECK_CUDA_ERRORS(cudaMalloc(&mesh_data_->texture_map_device, 
    rgb_texture_map.total() * rgb_texture_map.elemSize()));
  CHECK_CUDA_ERRORS(cudaMemcpyAsync(mesh_data_->texture_map_device, rgb_texture_map.data, 
    rgb_texture_map.total() * rgb_texture_map.elemSize(), cudaMemcpyHostToDevice, cuda_stream_));
  CHECK_CUDA_ERRORS(cudaStreamSynchronize(cuda_stream_));
  return GXF_SUCCESS;
}

std::pair<Eigen::Vector3f, Eigen::Vector3f> MeshStorage::FindMinMaxVertex(const aiMesh* mesh) {
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

float MeshStorage::CalcMeshDiameter(const aiMesh* mesh) {
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

gxf_result_t MeshStorage::LoadMeshData(const std::string& mesh_file_path) {
    // Log the mesh file path
  GXF_LOG_DEBUG("[MeshStorage] Attempting to load mesh from path: %s", mesh_file_path.c_str());
  
  // Check if file exists
  if (!std::filesystem::exists(mesh_file_path)) {
    GXF_LOG_ERROR("[MeshStorage] Mesh file does not exist at path: %s", mesh_file_path.c_str());
    return GXF_FAILURE;
  }

  // Free the previous mesh data if it exists
  if (mesh_data_->mesh_vertices_device != nullptr) {
    mesh_data_->FreeMeshDeviceMemory();
  }
  mesh_data_->mesh_file_path = mesh_file_path;

  Assimp::Importer importer;
  const aiScene* scene = importer.ReadFile(
    mesh_file_path,
    aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);

  if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
    GXF_LOG_ERROR("[MeshStorage] Error while loading mesh file ERROR::ASSIMP:: %s", 
                  importer.GetErrorString());
    return GXF_FAILURE;
  }

  if (scene->mNumMeshes == 0) {
    GXF_LOG_ERROR("[MeshStorage] No mesh was found in the mesh file");
    return GXF_FAILURE;
  }

  const aiMesh* mesh = scene->mMeshes[0];

  auto min_max_vertex = FindMinMaxVertex(mesh);
  mesh_data_->mesh_model_center = (min_max_vertex.second + min_max_vertex.first) / 2.0;
  mesh_data_->min_vertex = min_max_vertex.first;
  mesh_data_->max_vertex = min_max_vertex.second;
  mesh_data_->mesh_diameter = CalcMeshDiameter(mesh);

  std::vector<float> vertices;
  std::vector<float> texcoords;
  std::vector<int32_t> mesh_faces;

  for (unsigned int v = 0; v < mesh->mNumVertices; v++) {
    vertices.push_back(mesh->mVertices[v].x - mesh_data_->mesh_model_center[0]);
    vertices.push_back(mesh->mVertices[v].y - mesh_data_->mesh_model_center[1]);
    vertices.push_back(mesh->mVertices[v].z - mesh_data_->mesh_model_center[2]);

    if (mesh->mTextureCoords[0]) {
      texcoords.push_back(mesh->mTextureCoords[0][v].x);
      texcoords.push_back(1 - mesh->mTextureCoords[0][v].y);
    }
  }

  for (unsigned int f = 0; f < mesh->mNumFaces; f++) {
    const aiFace& face = mesh->mFaces[f];
    if (face.mNumIndices == 3) {
      for (unsigned int i = 0; i < face.mNumIndices; i++) {
        mesh_faces.push_back(face.mIndices[i]);
      }
    } else {
      GXF_LOG_ERROR("Only triangle is supported, but the object face has %u vertices.", face.mNumIndices);
      return GXF_FAILURE;
    }
  }

  mesh_data_->num_vertices = vertices.size() / 3;
  mesh_data_->num_texcoords = texcoords.size() / 2;
  mesh_data_->num_faces = mesh_faces.size() / 3;

  if (mesh_data_->num_vertices == 0 || mesh_data_->num_faces == 0) {
    GXF_LOG_ERROR("[MeshStorage] Empty input from mesh file.");
    GXF_LOG_ERROR("  - Vertices: %d", mesh_data_->num_vertices);
    GXF_LOG_ERROR("  - Texture coordinates: %d", mesh_data_->num_texcoords);
    GXF_LOG_ERROR("  - Faces: %d", mesh_data_->num_faces);
    return GXF_FAILURE;
  }

  mesh_data_->mesh_vertices = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
    vertices.data(), mesh_data_->num_vertices, 3);

  // Allocate and copy device memory
  size_t faces_size = mesh_faces.size() * sizeof(int32_t);
  size_t texcoords_size = texcoords.size() * sizeof(float);
  size_t mesh_vertices_size = vertices.size() * sizeof(float);

  CHECK_CUDA_ERRORS(cudaMalloc(&mesh_data_->mesh_vertices_device, mesh_vertices_size));
  CHECK_CUDA_ERRORS(cudaMalloc(&mesh_data_->mesh_faces_device, faces_size));
  if (texcoords_size > 0) {
    CHECK_CUDA_ERRORS(cudaMalloc(&mesh_data_->texcoords_device, texcoords_size));
  }

  CHECK_CUDA_ERRORS(cudaMemcpyAsync(mesh_data_->mesh_vertices_device, vertices.data(), 
    mesh_vertices_size, cudaMemcpyHostToDevice, cuda_stream_));
  CHECK_CUDA_ERRORS(cudaMemcpyAsync(mesh_data_->mesh_faces_device, mesh_faces.data(), 
    faces_size, cudaMemcpyHostToDevice, cuda_stream_));
  if (mesh_data_->texcoords_device != nullptr) {
    CHECK_CUDA_ERRORS(cudaMemcpyAsync(
      mesh_data_->texcoords_device, texcoords.data(), texcoords.size() * sizeof(float), cudaMemcpyHostToDevice, cuda_stream_));
  } 
  CHECK_CUDA_ERRORS(cudaStreamSynchronize(cuda_stream_));

  // Log final processed data
  GXF_LOG_DEBUG("[MeshStorage] Processed mesh data:");
  GXF_LOG_DEBUG("  - Processed vertices: %d", mesh_data_->num_vertices);
  GXF_LOG_DEBUG("  - Processed faces: %d", mesh_data_->num_faces);
  GXF_LOG_DEBUG("  - Processed texture coordinates: %d", mesh_data_->num_texcoords);
  GXF_LOG_DEBUG("  - Mesh diameter: %f", mesh_data_->mesh_diameter);

  // Load texture data
  std::string texture_path_str;
  // Get material and texture information
  if (mesh->mMaterialIndex >= 0) {
    aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
    if (material) {
      aiString texture_path;
      if (material->GetTexture(aiTextureType_DIFFUSE, 0, &texture_path) == AI_SUCCESS) {
        // Convert texture path to absolute path if it's relative
        std::filesystem::path mesh_dir = std::filesystem::path(mesh_file_path).parent_path();
        std::filesystem::path tex_path = mesh_dir / texture_path.C_Str();
        texture_path_str = tex_path.string();
        GXF_LOG_INFO("[MeshStorage] Found texture path: %s", tex_path.string().c_str());
      } else {
        GXF_LOG_WARNING("[MeshStorage] No texture path found for mesh");
      }
    } else {
      GXF_LOG_WARNING("[MeshStorage] No material found for mesh");
    }
  } else {
    GXF_LOG_WARNING("[MeshStorage] No material index found for mesh");
  }

  // Only reload texture if the path has changed or if no texture was loaded before
  if (texture_path_str != mesh_data_->texture_path || mesh_data_->texture_map_device == nullptr
      || !mesh_data_->has_tex) {
    GXF_LOG_DEBUG("[MeshStorage] Texture path has changed, reloading texture");
    auto result = LoadTextureData(texture_path_str);
    if (result != GXF_SUCCESS) {
      GXF_LOG_ERROR("[MeshStorage] Failed to load texture");
      return result;
    }
  }

  return GXF_SUCCESS;

}

gxf_result_t MeshStorage::TryReloadMesh() {
  if(mesh_data_->mesh_file_path != mesh_file_path_.get()) {
    auto result = LoadMeshData(mesh_file_path_.get());
    if (result != GXF_SUCCESS) {
        GXF_LOG_ERROR("[MeshStorage] Failed to load mesh file");
        return result;
    }
  }
  return GXF_SUCCESS;
}

gxf_result_t MeshStorage::deinitialize() {
  if (mesh_data_) {
    mesh_data_->FreeMeshDeviceMemory();
    mesh_data_->FreeTextureDeviceMemory();
    mesh_data_.reset();
  }
  return GXF_SUCCESS;
}

std::shared_ptr<const MeshData> MeshStorage::GetMeshData() const {
  return mesh_data_;
}

}  // namespace isaac_ros
}  // namespace nvidia