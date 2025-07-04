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

#ifndef NVIDIA_ISAAC_ROS_EXTENSIONS_MESH_STORAGE_HPP_
#define NVIDIA_ISAAC_ROS_EXTENSIONS_MESH_STORAGE_HPP_

#pragma once

#include <memory>
#include <string>
#include <utility>

#include <Eigen/Dense>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <opencv2/opencv.hpp>

#include "gxf/cuda/cuda_stream.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"
#include "gxf/core/component.hpp"
#include "gxf/core/handle.hpp"
#include "gxf/core/parameter.hpp"
#include "gxf/core/parameter_parser_std.hpp"

#include "foundationpose_utils.hpp"

namespace nvidia {
namespace isaac_ros {

// Object mesh and texture data
struct MeshData {
  std::string texture_path;
  std::string mesh_file_path;

  // Host data
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mesh_vertices;
  float mesh_diameter;

  Eigen::Vector3f mesh_model_center;
  Eigen::Vector3f min_vertex;
  Eigen::Vector3f max_vertex;
  
  // Device pointers for mesh
  float* mesh_vertices_device{nullptr};
  int32_t* mesh_faces_device{nullptr};
  float* texcoords_device{nullptr};
  
  // Device pointers for texture
  uint8_t* texture_map_device{nullptr};
  
  // Mesh metadata
  int num_vertices{0};
  int num_texcoords{0};
  int num_faces{0};
  
  bool has_tex{true};
  // Texture metadata
  int texture_map_height{0};
  int texture_map_width{0};
  int texture_map_channels{0};
  // Clean up device memory
  void FreeMeshDeviceMemory() {
    if (mesh_vertices_device) cudaFree(mesh_vertices_device);
    if (mesh_faces_device) cudaFree(mesh_faces_device);
    if (texcoords_device) cudaFree(texcoords_device);

    mesh_vertices_device = nullptr;
    mesh_faces_device = nullptr;
    texcoords_device = nullptr;
  }
  void FreeTextureDeviceMemory(){
    if (texture_map_device) cudaFree(texture_map_device);
    texture_map_device = nullptr;
  }

  ~MeshData() {
    FreeMeshDeviceMemory();
    FreeTextureDeviceMemory();
  }

};

// GXF component to load mesh and texture data, and provide interface to access
class MeshStorage : public gxf::Component {
 public:
  gxf_result_t registerInterface(gxf::Registrar* registrar)  override;
  gxf_result_t initialize()  override;
  gxf_result_t deinitialize()  override;

  // Public method to get mesh data
  std::shared_ptr<const MeshData> GetMeshData() const;

  // Public method to try to reload mesh
  gxf_result_t TryReloadMesh();

 private:
  gxf_result_t LoadTextureData(const std::string& texture_file_path);
  gxf_result_t LoadMeshData(const std::string& mesh_file_path);
  std::pair<Eigen::Vector3f, Eigen::Vector3f> FindMinMaxVertex(const aiMesh* mesh);
  float CalcMeshDiameter(const aiMesh* mesh);

  gxf::Parameter<std::string> mesh_file_path_;
  gxf::Parameter<gxf::Handle<gxf::CudaStreamPool>> cuda_stream_pool_;
  
  std::shared_ptr<MeshData> mesh_data_;
  cudaStream_t cuda_stream_;
  gxf::Handle<gxf::CudaStream> cuda_stream_handle_;
};

}  // namespace isaac_ros
}  // namespace nvidia 

#endif  // NVIDIA_ISAAC_ROS_EXTENSIONS_MESH_STORAGE_HPP_