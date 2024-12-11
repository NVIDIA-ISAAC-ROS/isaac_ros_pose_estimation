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

#include "foundationpose_render.hpp"
#include "foundationpose_utils.hpp"

#include <Eigen/Dense>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <string>

#include <cuda_runtime.h>
#include <cvcuda/OpFlip.hpp>
#include <cvcuda/OpWarpPerspective.hpp>
#include <cvcuda/OpConvertTo.hpp>

#include "gxf/multimedia/camera.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace isaac_ros {

namespace {

constexpr char kNamePoints[] = "points";
constexpr char kNamePoses[] = "poses";

constexpr char kRenderedTensor[] = "input_tensor1";
constexpr char kOriginalTensor[] = "input_tensor2";
constexpr char RAW_CAMERA_MODEL_GXF_NAME[] = "intrinsics";

constexpr size_t kVertexPoints = 3;
constexpr size_t kTriangleVertices = 3;
constexpr size_t kTexcoordsDim = 2;
constexpr size_t kPTMatrixDim = 3;
constexpr size_t kTransformationMatrixSize = 4;
constexpr size_t kPoseMatrixLength = 4;
constexpr size_t kNumChannels = 3;
constexpr size_t kOutputRank = 4;
// Number of slices on the input
constexpr int kNumBatches = 6;
constexpr int kFixTextureMapWidth = 1920;
constexpr int kFixTextureMapHeight = 1080;
constexpr int kFixTextureMapColor = 128;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMajorMatrix;

// From OpenCV camera (cvcam) coordinate system to the OpenGL camera (glcam) coordinate system
const Eigen::Matrix4f kGLCamInCVCam =
    (Eigen::Matrix4f(4, 4) << 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1).finished();

RowMajorMatrix ComputeTF(
    float left, float right, float top, float bottom, Eigen::Vector2i out_size) {
  left = std::round(left);
  right = std::round(right);
  top = std::round(top);
  bottom = std::round(bottom);

  RowMajorMatrix tf = Eigen::MatrixXf::Identity(3, 3);
  tf(0, 2) = -left;
  tf(1, 2) = -top;

  RowMajorMatrix new_tf = Eigen::MatrixXf::Identity(3, 3);
  new_tf(0, 0) = out_size(0) / (right - left);
  new_tf(1, 1) = out_size(1) / (bottom - top);

  auto result = new_tf * tf;
  return result;
}

std::vector<RowMajorMatrix> ComputeCropWindowTF(
    const std::vector<Eigen::MatrixXf>& poses, const Eigen::MatrixXf& K, Eigen::Vector2i out_size,
    float crop_ratio, float mesh_diameter) {
  // Compute the tf batch from the left, right, top, and bottom coordinates
  int B = poses.size();
  float r = mesh_diameter * crop_ratio / 2;
  Eigen::MatrixXf offsets(5, 3);
  offsets << 0, 0, 0, r, 0, 0, -r, 0, 0, 0, r, 0, 0, -r, 0;

  std::vector<RowMajorMatrix> tfs;
  for (int i = 0; i < B; i++) {
    auto block = poses[i].block<3, 1>(0, 3).transpose();
    Eigen::MatrixXf pts = block.replicate(offsets.rows(), 1).array() + offsets.array();
    Eigen::MatrixXf projected = (K * pts.transpose()).transpose();
    Eigen::MatrixXf uvs =
        projected.leftCols(2).array() / projected.rightCols(1).replicate(1, 2).array();
    Eigen::MatrixXf center = uvs.row(0);

    float radius = std::abs((uvs - center.replicate(uvs.rows(), 1)).rightCols(1).maxCoeff());
    float left = center(0, 0) - radius;
    float right = center(0, 0) + radius;
    float top = center(0, 1) - radius;
    float bottom = center(0, 1) + radius;

    tfs.push_back(ComputeTF(left, right, top, bottom, out_size));
  }
  return tfs;
}

// 2D tensor * 3D tensor
gxf_result_t TransformPts(
    std::vector<RowMajorMatrix>& output, const Eigen::MatrixXf& pts, const std::vector<Eigen::MatrixXf>& tfs) {
  // Get the dimensions of the inputs
  int rows = pts.rows();
  int cols = pts.cols();
  int tfs_size = tfs.size();
  if (tfs_size == 0) {
    GXF_LOG_ERROR("[FoundationposeRender] The transfomation matrix is empty! ");
    return GXF_FAILURE;
  }

  if (tfs[0].cols() != tfs[0].rows()) {
    GXF_LOG_ERROR("[FoundationposeRender] The transfomation matrix has different rows and cols! ");
    return GXF_FAILURE;
  }
  int dim = tfs[0].rows();

  if (cols != dim - 1) {
    GXF_LOG_ERROR("[FoundationposeRender] The dimension of pts and tf are not match! ");
    return GXF_FAILURE;
  }

  for (int i = 0; i < tfs_size; i++) {
    RowMajorMatrix transformed_matrix;
    transformed_matrix.resize(rows, dim - 1);
    auto submatrix = tfs[i].block(0, 0, dim - 1, dim - 1);
    auto last_col = tfs[i].block(0, dim - 1, dim - 1, 1);

    // Apply the transformation to the points
    for (int j = 0; j < rows; j++) {
      auto new_row = submatrix * pts.row(j).transpose() + last_col;
      transformed_matrix.row(j) = new_row.transpose();
    }
    output.push_back(transformed_matrix);
  }

  // Return the result vector
  return GXF_SUCCESS;
}

// Adding one column of ones to the matrix
Eigen::MatrixXf ToHomo(const Eigen::MatrixXf& pts) {
  int rows = pts.rows();
  int cols = pts.cols();

  Eigen::MatrixXf ones = Eigen::MatrixXf::Ones(rows, 1);

  Eigen::MatrixXf homo(rows, cols + 1);
  homo << pts, ones;

  return homo;
}

void WrapImgPtrToNHWCTensor(
    uint8_t* input_ptr, nvcv::Tensor& output_tensor, int N, int H, int W, int C) {
  nvcv::TensorDataStridedCuda::Buffer output_buffer;
  output_buffer.strides[3] = sizeof(uint8_t);
  output_buffer.strides[2] = C * output_buffer.strides[3];
  output_buffer.strides[1] = W * output_buffer.strides[2];
  output_buffer.strides[0] = H * output_buffer.strides[1];
  output_buffer.basePtr = reinterpret_cast<NVCVByte*>(input_ptr);

  nvcv::TensorShape::ShapeType shape{N, H, W, C};
  nvcv::TensorShape tensor_shape{shape, "NHWC"};
  nvcv::TensorDataStridedCuda output_data(tensor_shape, nvcv::TYPE_U8, output_buffer);
  output_tensor = nvcv::TensorWrapData(output_data);
}

void WrapFloatPtrToNHWCTensor(
    float* input_ptr, nvcv::Tensor& output_tensor, int N, int H, int W, int C) {
  nvcv::TensorDataStridedCuda::Buffer output_buffer;
  output_buffer.strides[3] = sizeof(float);
  output_buffer.strides[2] = C * output_buffer.strides[3];
  output_buffer.strides[1] = W * output_buffer.strides[2];
  output_buffer.strides[0] = H * output_buffer.strides[1];
  output_buffer.basePtr = reinterpret_cast<NVCVByte*>(input_ptr);

  nvcv::TensorShape::ShapeType shape{N, H, W, C};
  nvcv::TensorShape tensor_shape{shape, "NHWC"};
  nvcv::TensorDataStridedCuda output_data(tensor_shape, nvcv::TYPE_F32, output_buffer);
  output_tensor = nvcv::TensorWrapData(output_data);
}

gxf_result_t ProjectMatrixFromIntrinsics(
    Eigen::Matrix4f& proj_output, const Eigen::Matrix3f& K, int height, int width, float znear = 0.1, float zfar = 100.0,
    std::string window_coords = "y_down") {
  int x0 = 0;
  int y0 = 0;
  int w = width;
  int h = height;
  float nc = znear;
  float fc = zfar;

  float depth = fc - nc;
  float q = -(fc + nc) / depth;
  float qn = -2 * (fc * nc) / depth;

  // Get the projection matrix from camera K matrix
  if (window_coords == "y_up") {
    proj_output << 2 * K(0, 0) / w, -2 * K(0, 1) / w, (-2 * K(0, 2) + w + 2 * x0) / w, 0, 0,
        -2 * K(1, 1) / h, (-2 * K(1, 2) + h + 2 * y0) / h, 0, 0, 0, q, qn, 0, 0, -1, 0;
  } else if (window_coords == "y_down") {
    proj_output << 2 * K(0, 0) / w, -2 * K(0, 1) / w, (-2 * K(0, 2) + w + 2 * x0) / w, 0, 0,
        2 * K(1, 1) / h, (2 * K(1, 2) - h + 2 * y0) / h, 0, 0, 0, q, qn, 0, 0, -1, 0;
  } else {
    GXF_LOG_ERROR("[FoundationposeRender] The window coordinates should be y_up or y_down");
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

gxf_result_t ConstructBBox2D(
    Eigen::MatrixXf& bbox2d, const std::vector<RowMajorMatrix>& tfs, int H, int W) {
  Eigen::MatrixXf bbox2d_crop(2, 2);
  bbox2d_crop << 0.0, 0.0, W - 1, H - 1;

  std::vector<Eigen::MatrixXf> inversed_tfs;
  // Inverse tfs before transform
  for (size_t i = 0; i < tfs.size(); i++) {
    inversed_tfs.push_back(tfs[i].inverse());
  }

  std::vector<RowMajorMatrix> bbox2d_ori_vec;
  auto gxf_result_t = TransformPts(bbox2d_ori_vec, bbox2d_crop, inversed_tfs);
  if(gxf_result_t != GXF_SUCCESS) {
    GXF_LOG_ERROR("[FoundationposeRender] Failed to transform the 2D bounding box");
    return gxf_result_t;
  }

  for (size_t i = 0; i < bbox2d_ori_vec.size(); i++) {
    bbox2d.row(i) =
        Eigen::Map<Eigen::RowVectorXf>(bbox2d_ori_vec[i].data(), bbox2d_ori_vec[i].size());
  }
  return GXF_SUCCESS;
}

}  // namespace

gxf_result_t FoundationposeRender::registerInterface(gxf::Registrar* registrar) noexcept {
  gxf::Expected<void> result;

  result &= registrar->parameter(
      pose_receiver_, "pose_array_input", "Pose Array Input",
      "The pose estimation input as tensor ");

  result &= registrar->parameter(
      point_cloud_receiver_, "point_cloud_input", "Point Cloud Input",
      "The point cloud input as a tensor");

  result &= registrar->parameter(
      rgb_receiver_, "rgb_input", "RGB image Input", "The RGB image input as a videobuffer");

  result &= registrar->parameter(
      camera_model_receiver_, "camera_model_input", "Camera model input",
      "The camera intrinsics and extrinsics wrapped in a gxf message");

  result &= registrar->parameter(
      pose_array_transmitter_, "output", "Pose Array Output", "The ouput poses as a pose array");
  
  result &= registrar->parameter(
      mesh_file_path_, "mesh_file_path", "obj mesh file path", "Path to your object mesh file");

  result &= registrar->parameter(
      texture_path_, "texture_path", "texture file path", "Path to your texture file");

  result &= registrar->parameter(
      mode_, "mode", "Render Mode", "render mode select from refine or score");

  result &= registrar->parameter(
      crop_ratio_, "crop_ratio", "crop ratio", "Input image crop ratio");

  result &= registrar->parameter(
      resized_image_height_, "resized_image_height", "desired height", "Desired height of the rendered object");

  result &= registrar->parameter(
      resized_image_width_, "resized_image_width", "desired width", "Desired width of the rendered object");

  result &= registrar->parameter(
      min_depth_, "min_depth", "Minimum Depth", "Minimum allowed Z-axis value of pointcloud");

  result &= registrar->parameter(
      max_depth_, "max_depth", "Maximum Depth", "Maximum allowed  X,Y,Z-axis value of pointcloud to threshold");

  result &= registrar->parameter(
      refine_iterations_, "refine_iterations", "refine iterations", 
      "Number of iterations on the refine network (render->refine->transoformation)", 1);

  result &= registrar->parameter(
      allocator_, "allocator", "Allocator", "Output Allocator");

  result &= registrar->parameter(
      cuda_stream_pool_, "cuda_stream_pool", "Cuda Stream Pool",
      "Instance of gxf::CudaStreamPool to allocate CUDA stream.");

  // Only used when multiple refinement iterations are required
  result &= registrar->parameter(
      iterative_pose_receiver_, "iterative_pose_array_input", "Iterative Pose Array Input",
      "The pose estimation input as tensor",
      gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);

  result &= registrar->parameter(
      iterative_point_cloud_receiver_, "iterative_point_cloud_input", "Iterative Point Cloud Input",
      "The point cloud input as a tensor",
      gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);

  result &= registrar->parameter(
      iterative_rgb_receiver_, "iterative_rgb_input", "Iterative RGB image Input",
      "The RGB image input as a videobuffer",
      gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);

  result &= registrar->parameter(
      iterative_camera_model_receiver_, "iterative_camera_model_input", "Iterative Camera model input",
      "The camera intrinsics and extrinsics wrapped in a gxf message",
      gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);

  result &= registrar->parameter(
      iterative_point_cloud_transmitter_, "iterative_point_cloud_output", "Iterative Point Cloud Output",
      "The ouput poses as a pose array",
      gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  
  result &= registrar->parameter(
      iterative_rgb_transmitter_, "iterative_rgb_output", "Iterative RGB Output",
      "The ouput poses as a pose array",
      gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  
  result &= registrar->parameter(
      iterative_camera_model_transmitter_, "iterative_camera_model_output", "Iterative Camera Model Output",
      "The ouput poses as a pose array",
      gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);

  return gxf::ToResultCode(result);
}

gxf_result_t FoundationposeRender::start() noexcept {
  GXF_LOG_DEBUG("[FoundationposeRender] FoundationposeRender start");
  // Validate input parameters
  if (refine_iterations_ < 1) {
    GXF_LOG_ERROR("[FoundationposeRender] Refine iterations should be at least 1");
    return GXF_FAILURE;
  }

  // Load the obj mesh file
  // TODO: optimize the mesh loading by passing mesh data between codelets to avoid
  // duplicate calculation. Could calcuate from sampling node and broadcast to the other nodes
  Assimp::Importer importer;
  const aiScene* scene = importer.ReadFile(
      mesh_file_path_.get(),
      aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);

  if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
    GXF_LOG_ERROR("[FoundationposeRender] Error while loading mesh file ERROR::ASSIMP::");
    GXF_LOG_ERROR(importer.GetErrorString());
    return GXF_FAILURE;
  }

  if (scene->mNumMeshes == 0) {
    GXF_LOG_ERROR("[FoundationposeRender] No mesh was found in the mesh file");
    return GXF_FAILURE;
  }

  // Only take the first mesh
  const aiMesh* mesh = scene->mMeshes[0];

  mesh_diameter_ = CalcMeshDiameter(mesh);
  auto min_max_vertex = FindMinMaxVertex(mesh);
  auto mesh_model_center = (min_max_vertex.second + min_max_vertex.first) / 2.0;

  // Walk through each of the mesh's vertices
  for (unsigned int v = 0; v < mesh->mNumVertices; v++) {
    vertices_.push_back(mesh->mVertices[v].x - mesh_model_center[0]);
    vertices_.push_back(mesh->mVertices[v].y - mesh_model_center[1]);
    vertices_.push_back(mesh->mVertices[v].z - mesh_model_center[2]);

    // Check if the mesh has texture coordinates
    if (mesh->mTextureCoords[0]) {
      texcoords_.push_back(mesh->mTextureCoords[0][v].x);
      texcoords_.push_back(1 - mesh->mTextureCoords[0][v].y);
    }
  }

  // Walk through each of the mesh's faces (a face is a mesh its triangle)
  for (unsigned int f = 0; f < mesh->mNumFaces; f++) {
    const aiFace& face = mesh->mFaces[f];

    // We assume the face is a triangle due to aiProcess_Triangulate
    if (face.mNumIndices == 3) {
      for (unsigned int i = 0; i < face.mNumIndices; i++) {
        mesh_faces_.push_back(face.mIndices[i]);
      }
    } else {
      GXF_LOG_ERROR(
          "Only triangle is supported, but the object face has %u vertices. ", face.mNumIndices);
      return GXF_FAILURE;
    }
  }

  cv::Mat rgb_texture_map;
  if (!std::filesystem::exists(texture_path_.get())) {
    GXF_LOG_WARNING("[FoundationposeRender], %s could not be found, assign texture map with pure color",
                    texture_path_.get().c_str());
    rgb_texture_map = cv::Mat(kFixTextureMapHeight, kFixTextureMapWidth, CV_8UC3,
                              cv::Scalar(kFixTextureMapColor, kFixTextureMapColor, kFixTextureMapColor));
  } else {
    // Load the texture map
    cv::Mat texture_map = cv::imread(texture_path_.get());
    cv::cvtColor(texture_map, rgb_texture_map, cv::COLOR_BGR2RGB);
  }

  if (!rgb_texture_map.isContinuous()) {
    GXF_LOG_ERROR("[FoundationposeRender] Texture map is not continuous");
    return GXF_FAILURE;
  }

  if (rgb_texture_map.channels() != kNumChannels) {
    GXF_LOG_ERROR(
        "[FoundationposeRender] Recieved texture map has %d number of channels, but expected %lu.",
        rgb_texture_map.channels(), kNumChannels);
    return GXF_FAILURE;
  }
  texture_map_height_ = rgb_texture_map.rows;
  texture_map_width_ = rgb_texture_map.cols;
  GXF_LOG_DEBUG("[FoundationposeRender] Texture map height: %d, width: %d", texture_map_height_, texture_map_width_);

  // The number of vertices is the size of the vertices array divided by 3 (since it's x,y,z)
  num_vertices_ = vertices_.size() / kVertexPoints;
  // The number of texture coordinates is the size of the texcoords array divided by 2 (since it's u,v)
  num_texcoords_ = texcoords_.size() / kTexcoordsDim;
  // The number of faces is the size of the faces array divided by 3 (since each face has 3 edges)
  num_faces_ = mesh_faces_.size() / kTriangleVertices;

  // Check if the mesh data is valid
  if (num_vertices_ == 0 || num_texcoords_ == 0 || num_faces_ == 0) {
    GXF_LOG_ERROR("[FoundationposeRender] Empty input from mesh file. ");
    return GXF_FAILURE;
  }

  if (sizeof(vertices_[0]) != sizeof(float) || 
      sizeof(texcoords_[0]) != sizeof(float) || 
      sizeof(mesh_faces_[0]) != sizeof(int32_t)) {  
    GXF_LOG_ERROR("[FoundationposeRender] Invalid data type from mesh file. ");
    return GXF_FAILURE;
  }

  // Assign vertices to the class memeber
  mesh_vertices_ =
      Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
          vertices_.data(), num_vertices_, 3);

  GXF_LOG_DEBUG("[FoundationposeRender] Number of mesh vertices: %d", num_vertices_);
  GXF_LOG_DEBUG("[FoundationposeRender] Number of mesh faces: %d", num_faces_);
  GXF_LOG_DEBUG("[FoundationposeRender] Number of mesh texcoords: %d", num_texcoords_);

  // Get cuda stream from stream pool
  auto maybe_stream = cuda_stream_pool_.get()->allocateStream();
  if (!maybe_stream) { return gxf::ToResultCode(maybe_stream); }

  cuda_stream_handle_ = std::move(maybe_stream.value());
  if (!cuda_stream_handle_->stream()) {
    GXF_LOG_ERROR("[FoundationposeRender] Allocated stream is not initialized!");
    return GXF_FAILURE;
  }
  if (!cuda_stream_handle_.is_null()) {
    cuda_stream_ = cuda_stream_handle_->stream().value();
  }

  // Allocate device memory for mesh data
  size_t faces_size = mesh_faces_.size() * sizeof(int32_t);
  size_t texcoords_size = texcoords_.size() * sizeof(float);
  size_t mesh_vertices_size = vertices_.size() * sizeof(float);
  
  CHECK_CUDA_ERRORS(cudaMalloc(&mesh_vertices_device_, mesh_vertices_size));
  CHECK_CUDA_ERRORS(cudaMalloc(&mesh_faces_device_, faces_size));
  CHECK_CUDA_ERRORS(cudaMalloc(&texcoords_device_, texcoords_size));
  CHECK_CUDA_ERRORS(cudaMalloc(&texture_map_device_, rgb_texture_map.total() * kNumChannels));

  CHECK_CUDA_ERRORS(cudaMemcpyAsync(
    mesh_vertices_device_, vertices_.data(), mesh_vertices_size , cudaMemcpyHostToDevice, cuda_stream_));
  CHECK_CUDA_ERRORS(cudaMemcpyAsync(
    mesh_faces_device_, mesh_faces_.data(), faces_size, cudaMemcpyHostToDevice, cuda_stream_));
  CHECK_CUDA_ERRORS(cudaMemcpyAsync(
    texcoords_device_, texcoords_.data(), texcoords_.size() * sizeof(float), cudaMemcpyHostToDevice, cuda_stream_));
  CHECK_CUDA_ERRORS(cudaMemcpyAsync(
    texture_map_device_, reinterpret_cast<uint8_t*>(rgb_texture_map.data),
    rgb_texture_map.total() * kNumChannels, cudaMemcpyHostToDevice, cuda_stream_));

  // Preprocess mesh data
  nvcv::Tensor texture_map_tensor;
  WrapImgPtrToNHWCTensor(texture_map_device_, texture_map_tensor, 1, texture_map_height_, texture_map_width_, kNumChannels);

  nvcv::TensorShape::ShapeType shape{1, texture_map_height_, texture_map_width_, kNumChannels};
  nvcv::TensorShape tensor_shape{shape, "NHWC"};
  float_texture_map_tensor_ = nvcv::Tensor(tensor_shape, nvcv::TYPE_F32);

  const float scale_factor =  1.0f / 255.0f;
  cvcuda::ConvertTo convert_op;
  convert_op(cuda_stream_, texture_map_tensor, float_texture_map_tensor_, scale_factor, 0.0f);
  CHECK_CUDA_ERRORS(cudaGetLastError());
  CHECK_CUDA_ERRORS(cudaStreamSynchronize(cuda_stream_));

  return GXF_SUCCESS;
}


gxf_result_t FoundationposeRender::NvdiffrastRender(
    cudaStream_t cuda_stream, std::vector<Eigen::MatrixXf>& poses, Eigen::Matrix3f& K, Eigen::MatrixXf& bbox2d, int rgb_H,
    int rgb_W, int H, int W, nvcv::Tensor& flip_color_tensor, nvcv::Tensor& flip_xyz_map_tensor) {
  // N represents the number of poses
  size_t N = poses.size();

  // For every pose, transform the vertices to the camera space
  if (kVertexPoints != kPoseMatrixLength - 1) {
    GXF_LOG_ERROR("[FoundationposeRender] The vertice channel should be same as pose matrix length - 1");
    return GXF_FAILURE;
  }
  transform_pts(cuda_stream, pts_cam_device_, mesh_vertices_device_, pose_device_, num_vertices_, kVertexPoints, N, kPoseMatrixLength);
  CHECK_CUDA_ERRORS(cudaGetLastError());

  Eigen::Matrix4f projection_mat;
  auto gxf_result = ProjectMatrixFromIntrinsics(projection_mat, K, rgb_H, rgb_W);
  if (gxf_result != GXF_SUCCESS) {
    return GXF_FAILURE;
  }

  // Homogeneliaze the vertices to N * 4
  auto pose_homo = ToHomo(mesh_vertices_);
  if (pose_homo.rows() != num_vertices_) {
    GXF_LOG_ERROR("[FoundationposeRender] The number of vertice should not change after homogeneliaze ");
    return GXF_FAILURE;
  }
  if (pose_homo.cols() != mesh_vertices_.cols() + 1) {
    GXF_LOG_ERROR("[FoundationposeRender] Points per vertex should increase by one after homogeneliaze");
    return GXF_FAILURE;
  }

  // Allocate device memory for the intermedia results on the first frame
  size_t pose_clip_size = N * num_vertices_ * 4 * sizeof(float);
  size_t rast_out_size = N * H * W * 4 * sizeof(float);
  size_t color_size = N * H * W * kNumChannels * sizeof(float);
  size_t xyz_map_size = N * H * W * kNumChannels * sizeof(float);
  size_t texcoords_out_size = N * H * W * kTexcoordsDim * sizeof(float);
  size_t bbox2d_size = N * 4 * sizeof(float);

  if (render_data_cached_ == false) {
    CHECK_CUDA_ERRORS(cudaMalloc(&pose_clip_device_, pose_clip_size));
    CHECK_CUDA_ERRORS(cudaMalloc(&rast_out_device_, rast_out_size));
    CHECK_CUDA_ERRORS(cudaMalloc(&texcoords_out_device_, texcoords_out_size));
    CHECK_CUDA_ERRORS(cudaMalloc(&color_device_, color_size));
    CHECK_CUDA_ERRORS(cudaMalloc(&xyz_map_device_, xyz_map_size));
    CHECK_CUDA_ERRORS(cudaMalloc(&pose_clip_device_, pose_clip_size));
    CHECK_CUDA_ERRORS(cudaMalloc(&bbox2d_device_, bbox2d_size));
    cr_ = new CR::CudaRaster();
    render_data_cached_ = true;
  }

  std::vector<float> h_bbox2d;
  for(int j=0;j<bbox2d.rows();j++){
      for(int k=0;k<bbox2d.cols();k++){
          h_bbox2d.push_back(bbox2d(j,k));
      }
  }

  CHECK_CUDA_ERRORS(cudaMemcpyAsync(bbox2d_device_, h_bbox2d.data(), bbox2d_size, cudaMemcpyHostToDevice, cuda_stream));
  generate_pose_clip(cuda_stream, pose_clip_device_, pose_device_, bbox2d_device_, mesh_vertices_device_, projection_mat, rgb_H, rgb_W, num_vertices_, N);
  CHECK_CUDA_ERRORS(cudaGetLastError());

  rasterize(
      cuda_stream, cr_,
      pose_clip_device_, mesh_faces_device_, rast_out_device_,
      num_vertices_, num_faces_,
      H, W, N);
  CHECK_CUDA_ERRORS(cudaGetLastError());

  interpolate(
      cuda_stream,
      pts_cam_device_, rast_out_device_, mesh_faces_device_, xyz_map_device_,
      num_vertices_, num_faces_, kVertexPoints,
      H, W, N);
  CHECK_CUDA_ERRORS(cudaGetLastError());

  interpolate(
      cuda_stream,
      texcoords_device_, rast_out_device_, mesh_faces_device_, texcoords_out_device_,
      num_vertices_, num_faces_, kTexcoordsDim,
      H, W, N);
  CHECK_CUDA_ERRORS(cudaGetLastError());

  auto float_texture_map_data = float_texture_map_tensor_.exportData<nvcv::TensorDataStridedCuda>();
  texture(
      cuda_stream,
      reinterpret_cast<float*>(float_texture_map_data->basePtr()),
      texcoords_out_device_,
      color_device_,
      texture_map_height_, texture_map_width_, kNumChannels,
      1, H, W, N);
  CHECK_CUDA_ERRORS(cudaGetLastError());

  float min_value = 0.0;
  float max_value = 1.0;
  clamp(cuda_stream, color_device_, min_value, max_value, N * H * W * kNumChannels);
  CHECK_CUDA_ERRORS(cudaGetLastError());

  nvcv::Tensor color_tensor, xyz_map_tensor;
  WrapFloatPtrToNHWCTensor(color_device_, color_tensor, N, H, W, kNumChannels);
  WrapFloatPtrToNHWCTensor(xyz_map_device_, xyz_map_tensor, N, H, W, kNumChannels);

  cvcuda::Flip flip_op;
  flip_op(cuda_stream, color_tensor, flip_color_tensor, 0);
  CHECK_CUDA_ERRORS(cudaGetLastError());

  flip_op(cuda_stream, xyz_map_tensor, flip_xyz_map_tensor, 0);
  CHECK_CUDA_ERRORS(cudaGetLastError());

  return GXF_SUCCESS;
}

gxf_result_t FoundationposeRender::tick() noexcept {
  GXF_LOG_DEBUG("[FoundationposeRender] tick");

  // Selective receive messages based on the states
  gxf::Entity rgb_message;
  gxf::Entity camera_model_message;
  gxf::Entity pose_message;
  gxf::Entity xyz_message;

  // Receive from upstream node on the first iteration
  if (iteration_count_ == 0) {
    auto maybe_rgb_message = rgb_receiver_->receive();
    if (!maybe_rgb_message) {
      GXF_LOG_ERROR("[FoundationposeRender] Failed to receive image message");
      return maybe_rgb_message.error();
    }
    rgb_message = maybe_rgb_message.value();

    auto maybe_camera_model_message = camera_model_receiver_->receive();
    if (!maybe_camera_model_message) {
      GXF_LOG_ERROR("[FoundationposeRender] Failed to receive camera model message");
      return maybe_camera_model_message.error();
    }
    camera_model_message = maybe_camera_model_message.value();
  
    auto maybe_pose_message = pose_receiver_->receive();
    if (!maybe_pose_message) {
      GXF_LOG_ERROR("[FoundationposeRender] Failed to receive image message");
      return maybe_pose_message.error();
    }
    pose_message = maybe_pose_message.value();

    auto maybe_xyz_message = point_cloud_receiver_->receive();
    if (!maybe_xyz_message) {
      GXF_LOG_ERROR("[FoundationposeRender] Failed to receive image message");
      return maybe_xyz_message.error();
    }
    xyz_message = maybe_xyz_message.value();;
  } else {
    // Receive the inputs published by itself
    auto iterative_rgb_receiver = iterative_rgb_receiver_.try_get();
    if (!iterative_rgb_receiver) {
      GXF_LOG_ERROR("[FoundationposeRender] Failed to get iterative RGB receiver.");
      return gxf::ToResultCode(iterative_rgb_receiver);
    }
    auto maybe_rgb_message = iterative_rgb_receiver.value()->receive();
    if (!maybe_rgb_message) {
      GXF_LOG_ERROR("[FoundationposeRender] Failed to receive image message");
      return maybe_rgb_message.error();
    }
    rgb_message = maybe_rgb_message.value();

    auto iterative_camera_model_receiver = iterative_camera_model_receiver_.try_get();
    if (!iterative_camera_model_receiver) {
      GXF_LOG_ERROR("[FoundationposeRender] Failed to get iterative camera model receiver.");
      return gxf::ToResultCode(iterative_camera_model_receiver);
    }
    auto maybe_camera_model_message = iterative_camera_model_receiver.value()->receive();
    if (!maybe_camera_model_message) {
      GXF_LOG_ERROR("[FoundationposeRender] Failed to receive camera model message");
      return maybe_camera_model_message.error();
    }
    camera_model_message = maybe_camera_model_message.value();

    auto iterative_point_cloud_receiver = iterative_point_cloud_receiver_.try_get();
    if (!iterative_point_cloud_receiver) {
      GXF_LOG_ERROR("[FoundationposeRender] Failed to get iterative camera model receiver.");
      return gxf::ToResultCode(iterative_point_cloud_receiver);
    }
    auto maybe_xyz_message = iterative_point_cloud_receiver.value()->receive();
    if (!maybe_xyz_message) {
      GXF_LOG_ERROR("[FoundationposeRender] Failed to receive image message");
      return maybe_xyz_message.error();
    }
    xyz_message = maybe_xyz_message.value();

    auto iterative_pose_receiver = iterative_pose_receiver_.try_get();
    if (!iterative_pose_receiver) {
      GXF_LOG_ERROR("[FoundationposeRender] Failed to get iterative pose receiver.");
      return gxf::ToResultCode(iterative_pose_receiver);
    }
    auto maybe_pose_message = iterative_pose_receiver.value()->receive();
    if (!maybe_pose_message) {
      GXF_LOG_ERROR("[FoundationposeRender] Failed to receive iterative pose message");
      return maybe_pose_message.error();
    }
    pose_message = maybe_pose_message.value();
  }

  auto maybe_rgb_image = rgb_message.get<gxf::VideoBuffer>();
  if (!maybe_rgb_image) {
    GXF_LOG_ERROR("[FoundationposeRender] Failed to get rgb image from message");
    return maybe_rgb_image.error();
  }

  auto maybe_gxf_camera_model = camera_model_message.get<nvidia::gxf::CameraModel>(
    RAW_CAMERA_MODEL_GXF_NAME);
  if (!maybe_gxf_camera_model) {
    GXF_LOG_ERROR("[FoundationposeRender] Failed to receive image message");
    return maybe_gxf_camera_model.error();
  }
  auto gxf_camera_model = maybe_gxf_camera_model.value();

  auto maybe_xyz_map = xyz_message.get<gxf::Tensor>(kNamePoints);
  if (!maybe_xyz_map) {
    GXF_LOG_ERROR("[FoundationposeRender] Failed to get xyz map from message");
    return maybe_xyz_map.error();
  }

  auto maybe_poses = pose_message.get<gxf::Tensor>(kNamePoses);
  if (!maybe_poses) {
    GXF_LOG_ERROR("[FoundationposeRender] Failed to get poses from message");
    return maybe_poses.error();
  }

  // Validate input
  auto rgb_img_handle = maybe_rgb_image.value();
  auto rgb_img_info = rgb_img_handle->video_frame_info();

  auto xyz_map_handle = maybe_xyz_map.value();
  auto poses_handle = maybe_poses.value();

  const size_t pose_nums = poses_handle->shape().dimension(0);
  const size_t pose_rows = poses_handle->shape().dimension(1);
  const size_t pose_cols = poses_handle->shape().dimension(2);

  if (pose_nums == 0) {
    GXF_LOG_ERROR("[FoundationposeRender] The received pose is empty");
    return GXF_FAILURE;
  }

  // Each pose should be a 4*4 matrix
  if (pose_rows != kPoseMatrixLength || pose_cols != kPoseMatrixLength) {
    GXF_LOG_ERROR("[FoundationposeRender] The received pose has wrong dimension");
    return GXF_FAILURE;
  }

  if (poses_handle->size() != pose_nums * pose_rows * pose_cols * sizeof(float)) {
    GXF_LOG_ERROR("[FoundationposeRender] Unexpected size of the pose tensor");
    return GXF_FAILURE;
  }

  const uint32_t rgb_H = rgb_img_info.height;
  const uint32_t rgb_W = rgb_img_info.width;

  const uint32_t N = pose_nums;
  const uint32_t W = resized_image_width_;
  const uint32_t H = resized_image_height_;
  const uint32_t C = kNumChannels;

  const uint32_t total_poses = N * kNumBatches;

  // Create Eigen matrix from the gxf camera model message
  Eigen::Matrix3f K;
  K << gxf_camera_model->focal_length.x, 0.0, gxf_camera_model->principal_point.x,
       0.0, gxf_camera_model->focal_length.y, gxf_camera_model->principal_point.y,
       0.0, 0.0, 1.0;

  // Malloc device memory for the output
  if (!rgb_data_cached_) {
    CHECK_CUDA_ERRORS(cudaMalloc(&pts_cam_device_,  N * num_vertices_ * kVertexPoints * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaMalloc(&transformed_xyz_map_device_, N * H * W * C * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaMalloc(&transformed_rgb_device_, N * H * W * C * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaMalloc(&score_rendered_output_device_, total_poses * 2 * H * W * C * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaMalloc(&score_original_output_device_, total_poses * 2 * H * W * C * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaMalloc(&wp_image_device_, N * H * W * C * sizeof(uint8_t)));
    CHECK_CUDA_ERRORS(cudaMalloc(&trans_matrix_device_,N*9*sizeof(float)));

    nvcv::TensorShape::ShapeType shape{N, H, W, C};
    nvcv::TensorShape tensor_shape{shape, "NHWC"};
    render_rgb_tensor_ = nvcv::Tensor(tensor_shape, nvcv::TYPE_F32);
    render_xyz_map_tensor_ = nvcv::Tensor(tensor_shape, nvcv::TYPE_F32);

    rgb_data_cached_ = true;
  }

  std::vector<float> pose_host(pose_nums * pose_rows * pose_cols);
  pose_device_ = reinterpret_cast<float*>(poses_handle->pointer());
  CHECK_CUDA_ERRORS(cudaMemcpyAsync(
      pose_host.data(), poses_handle->pointer(), poses_handle->size(), cudaMemcpyDeviceToHost, cuda_stream_));
  cudaStreamSynchronize(cuda_stream_);

  // Construct pose matrix from pointer
  std::vector<Eigen::MatrixXf> poses;
  for (size_t i = 0; i < pose_nums; i++) {
    Eigen::Map<Eigen::MatrixXf> mat(
        pose_host.data() + i * pose_rows * pose_cols, pose_rows, pose_cols);
    poses.push_back(mat);
  }
  
  Eigen::Vector2i out_size = {H, W};
  auto tfs = ComputeCropWindowTF(poses, K, out_size, crop_ratio_, mesh_diameter_);
  if (tfs.size() == 0) {
    GXF_LOG_ERROR("[FoundationposeRender] The transform matrix vector is empty");
    return GXF_FAILURE;
  }

  // Convert the bbox2d from vector N of 2*2 matrix into a N*4 matrix
  Eigen::MatrixXf bbox2d(tfs.size(), 4);
  auto gxf_result = ConstructBBox2D(bbox2d, tfs, H, W);
  if (gxf_result != GXF_SUCCESS) {
    return GXF_FAILURE;
  }

  // Render the object using give poses
  gxf_result = NvdiffrastRender(
      cuda_stream_, poses, K, bbox2d, rgb_H, rgb_W, H, W, render_rgb_tensor_, render_xyz_map_tensor_);
  if (gxf_result != GXF_SUCCESS) {
    return GXF_FAILURE;
  }

  // WarpPerspective on each image and xyz map, and cache the output for the future iterations
  nvcv::TensorShape::ShapeType rgb_shape{1, rgb_H, rgb_W, C};
  nvcv::TensorShape rgb_tensor_shape{rgb_shape, "NHWC"};

  nvcv::Tensor rgb_tensor = nvcv::Tensor(rgb_tensor_shape, nvcv::TYPE_U8);
  nvcv::Tensor xyz_map_tensor = nvcv::Tensor(rgb_tensor_shape, nvcv::TYPE_F32);

  WrapImgPtrToNHWCTensor(reinterpret_cast<uint8_t*>(rgb_img_handle->pointer()), rgb_tensor, 1, rgb_H, rgb_W, C);
  WrapFloatPtrToNHWCTensor(reinterpret_cast<float*>(xyz_map_handle->pointer()), xyz_map_tensor, 1, rgb_H, rgb_W, C);

  const int rgb_flags = NVCV_INTERP_LINEAR;
  const int xyz_flags = NVCV_INTERP_NEAREST;
  const float4 border_value = {0,0,0,0};

  const float scale_factor =  1.0f / 255.0f;
  int num_of_trans_mat = N;
  cvcuda::WarpPerspective warpPerspectiveOpBatch(num_of_trans_mat);
  cvcuda::ConvertTo convert_op;

  std::vector<float> trans_mat_flattened(N*9,0);
  for(size_t index = 0; index<N; index++){
    for (size_t i = 0; i < kPTMatrixDim; i++) {
        for (size_t j = 0; j < kPTMatrixDim; j++) {
          trans_mat_flattened[index*kPTMatrixDim*kPTMatrixDim+i*kPTMatrixDim+j] = tfs[index](i,j);
        }
      }
  }
  CHECK_CUDA_ERRORS(cudaMemcpyAsync(trans_matrix_device_,
                                    trans_mat_flattened.data(),
                                    trans_mat_flattened.size()*sizeof(float),
                                    cudaMemcpyHostToDevice, cuda_stream_));

  nvcv::TensorDataStridedCuda::Buffer buf_trans_mat;
  buf_trans_mat.strides[1] = sizeof(float);
  buf_trans_mat.strides[0] = 9*buf_trans_mat.strides[1];
  buf_trans_mat.basePtr    = reinterpret_cast<NVCVByte *>(trans_matrix_device_);

  auto transMatrixTensor = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
      nvcv::TensorShape({num_of_trans_mat, 9}, nvcv::TENSOR_NW),
      nvcv::TYPE_F32,
      buf_trans_mat
  });

  // Build rgb image batch input
  const nvcv::ImageFormat fmt_rgb = nvcv::FMT_RGB8;
  std::vector<nvcv::Image> wp_rgb_src;
  int imageSizeRGB = rgb_H * rgb_W * fmt_rgb.planePixelStrideBytes(0);
  nvcv::ImageDataStridedCuda::Buffer buf_rgb;
  buf_rgb.numPlanes           = 1;
  buf_rgb.planes[0].width     = rgb_W;
  buf_rgb.planes[0].height    = rgb_H;
  buf_rgb.planes[0].rowStride = rgb_W*fmt_rgb.planePixelStrideBytes(0);
  buf_rgb.planes[0].basePtr   = reinterpret_cast<NVCVByte *>(rgb_img_handle->pointer());
  auto img_rgb = nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{fmt_rgb, buf_rgb});

  for (int i = 0; i < num_of_trans_mat; ++i) {
      wp_rgb_src.emplace_back(img_rgb);
  }

  nvcv::ImageBatchVarShape batch_wp_rgb_src(num_of_trans_mat);
  batch_wp_rgb_src.pushBack(wp_rgb_src.begin(), wp_rgb_src.end());  

  // Build xyz map batch input
  const nvcv::ImageFormat fmt_xyz = nvcv::FMT_RGBf32;
  std::vector<nvcv::Image> wp_xyz_src;
  int imageSizeXYZ = rgb_H * rgb_W * fmt_xyz.planePixelStrideBytes(0);
  nvcv::ImageDataStridedCuda::Buffer buf_xyz;
  buf_xyz.numPlanes           = 1;
  buf_xyz.planes[0].width     = rgb_W;
  buf_xyz.planes[0].height    = rgb_H;
  buf_xyz.planes[0].rowStride = rgb_W*fmt_xyz.planePixelStrideBytes(0);
  buf_xyz.planes[0].basePtr   = reinterpret_cast<NVCVByte *>(xyz_map_handle->pointer());
  auto img_xyz = nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{fmt_xyz, buf_xyz});

  for (int i = 0; i < num_of_trans_mat; ++i) {
      wp_xyz_src.emplace_back(img_xyz);
  }

  nvcv::ImageBatchVarShape batch_wp_xyz_src(num_of_trans_mat);
  batch_wp_xyz_src.pushBack(wp_xyz_src.begin(), wp_xyz_src.end()); 

  // Build batched RGB output tensor
  std::vector<nvcv::Image> wp_rgb_dst;
  for (int i = 0; i < num_of_trans_mat; ++i) {        
      nvcv::ImageDataStridedCuda::Buffer buf;
      buf.numPlanes           = 1;
      buf.planes[0].width     = W;
      buf.planes[0].height    = H;
      buf.planes[0].rowStride = W * fmt_rgb.planePixelStrideBytes(0);
      buf.planes[0].basePtr   = reinterpret_cast<NVCVByte *>(wp_image_device_ + i * H * W * C);
      auto img = nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{fmt_rgb, buf});
      wp_rgb_dst.push_back(img);
  }
  nvcv::ImageBatchVarShape batch_wp_rgb_dst(num_of_trans_mat);
  batch_wp_rgb_dst.pushBack(wp_rgb_dst.begin(),wp_rgb_dst.end());

  // Build batched XYZ map output tensor
  std::vector<nvcv::Image> wp_xyz_dst;
  for (int i = 0; i < num_of_trans_mat; ++i) {        
        nvcv::ImageDataStridedCuda::Buffer buf;
        buf.numPlanes           = 1;
        buf.planes[0].width     = W;
        buf.planes[0].height    = H;
        buf.planes[0].rowStride = W * fmt_xyz.planePixelStrideBytes(0);
        buf.planes[0].basePtr   = reinterpret_cast<NVCVByte *>(transformed_xyz_map_device_ + i * H * W * C);
        auto img = nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{fmt_xyz, buf});
        wp_xyz_dst.push_back(img);
  }
  nvcv::ImageBatchVarShape batch_wp_xyz_dst(num_of_trans_mat);
  batch_wp_xyz_dst.pushBack(wp_xyz_dst.begin(),wp_xyz_dst.end());

  // Warp Perspective for RGB image and XYZ map
  warpPerspectiveOpBatch(cuda_stream_, batch_wp_rgb_src, batch_wp_rgb_dst, transMatrixTensor, rgb_flags, NVCV_BORDER_CONSTANT, border_value);
  CHECK_CUDA_ERRORS(cudaGetLastError());
  warpPerspectiveOpBatch(cuda_stream_, batch_wp_xyz_src, batch_wp_xyz_dst, transMatrixTensor, xyz_flags, NVCV_BORDER_CONSTANT, border_value);
  CHECK_CUDA_ERRORS(cudaGetLastError());

  // Convert RGB image from U8 to float
  nvcv::TensorShape::ShapeType transformed_shape{N,H,W,C};
  nvcv::TensorShape transformed_tensor_shape{transformed_shape, "NHWC"};

  nvcv::Tensor transformed_rgb_tensor = nvcv::Tensor(transformed_tensor_shape, nvcv::TYPE_U8);
  nvcv::Tensor float_rgb_tensor = nvcv::Tensor(transformed_tensor_shape, nvcv::TYPE_F32);

  WrapImgPtrToNHWCTensor(wp_image_device_, transformed_rgb_tensor, N, H, W, C);
  WrapFloatPtrToNHWCTensor(transformed_rgb_device_, float_rgb_tensor, N, H, W, C);
 
  convert_op(cuda_stream_, transformed_rgb_tensor, float_rgb_tensor, scale_factor, 0.0f);
  CHECK_CUDA_ERRORS(cudaGetLastError());

  threshold_and_downscale_pointcloud(
      cuda_stream_,
      transformed_xyz_map_device_,
      reinterpret_cast<float*>(poses_handle->pointer()),
      N, W * H, mesh_diameter_ / 2, min_depth_, max_depth_);
  CHECK_CUDA_ERRORS(cudaGetLastError());

  auto render_rgb_data = render_rgb_tensor_.exportData<nvcv::TensorDataStridedCuda>();
  auto render_xyz_map_data = render_xyz_map_tensor_.exportData<nvcv::TensorDataStridedCuda>();

  threshold_and_downscale_pointcloud(
      cuda_stream_,
      reinterpret_cast<float*>(render_xyz_map_data->basePtr()),
      reinterpret_cast<float*>(poses_handle->pointer()),
      N, W * H, mesh_diameter_ / 2, min_depth_, max_depth_);
  CHECK_CUDA_ERRORS(cudaGetLastError());

  // Score mode, accumulation stage
  // Only accumulate the output tensors and return
  if (mode_.get() == "score" && score_recevied_batches_ < kNumBatches - 1) {
    auto score_output_offset = score_recevied_batches_*N*H*W*2*C;
    concat(
        cuda_stream_,
        reinterpret_cast<float*>(render_rgb_data->basePtr()),
        reinterpret_cast<float*>(render_xyz_map_data->basePtr()),
        score_rendered_output_device_ + score_output_offset,
        N, H, W, C, C);
    CHECK_CUDA_ERRORS(cudaGetLastError());

    concat(
        cuda_stream_,
        transformed_rgb_device_,
        transformed_xyz_map_device_,
        score_original_output_device_ + score_output_offset,
        N, H, W, C, C);
    CHECK_CUDA_ERRORS(cudaGetLastError());
    CHECK_CUDA_ERRORS(cudaStreamSynchronize(cuda_stream_));
    score_recevied_batches_ += 1;
    return GXF_SUCCESS;
  }

  // Refine mode, or all input required by score inference are received.
  // Allocate output message
  auto maybe_output_message = gxf::Entity::New(context());
  if (!maybe_output_message) {
    GXF_LOG_ERROR("[FoundationposeRender] Failed to allocate PoseArray Message");
    return gxf::ToResultCode(maybe_output_message);
  }
  auto output_message = maybe_output_message.value();

  auto maybe_rendered_tensor = output_message.add<gxf::Tensor>(kRenderedTensor);
  if (!maybe_rendered_tensor) {
    GXF_LOG_ERROR("[FoundationposeRender] Failed to allocate ouptut Tensor 1");
    return gxf::ToResultCode(maybe_rendered_tensor);
  }
  auto rendered_tensor = maybe_rendered_tensor.value();

  auto maybe_original_tensor = output_message.add<gxf::Tensor>(kOriginalTensor);
  if (!maybe_original_tensor) {
    GXF_LOG_ERROR("[FoundationposeRender] Failed to allocate output Tensor 2");
    return gxf::ToResultCode(maybe_original_tensor);
  }
  auto original_tensor = maybe_original_tensor.value();

  auto maybe_added_timestamp = AddInputTimestampToOutput(output_message, rgb_message);
  if (!maybe_added_timestamp) {
    GXF_LOG_ERROR("[FoundationposeRender] Failed to add timestamp");
    return gxf::ToResultCode(maybe_added_timestamp);
  }

  uint32_t batch_size = mode_.get() == "refine" ? N : total_poses;
  std::array<int32_t, nvidia::gxf::Shape::kMaxRank> output_shape{
      static_cast<int>(batch_size), static_cast<int>(H), static_cast<int>(W), C + C};

  // Initializing GXF tensor
  auto result = rendered_tensor->reshape<float>(
      nvidia::gxf::Shape{output_shape, kOutputRank}, nvidia::gxf::MemoryStorageType::kDevice,
      allocator_);
  if (!result) {
    GXF_LOG_ERROR("[FoundationposeRender] Failed to reshape rendered Tensor");
    return gxf::ToResultCode(result);
  }

  result = original_tensor->reshape<float>(
      nvidia::gxf::Shape{output_shape, kOutputRank}, nvidia::gxf::MemoryStorageType::kDevice,
      allocator_);
  if (!result) {
    GXF_LOG_ERROR("[FoundationposeRender] Failed to reshape original tensor");
    return gxf::ToResultCode(result);
  }

  if (mode_.get() == "refine") {
    // Refine mode, concat the output publish
    concat(
      cuda_stream_,
      reinterpret_cast<float*>(render_rgb_data->basePtr()),
      reinterpret_cast<float*>(render_xyz_map_data->basePtr()),
      reinterpret_cast<float*>(rendered_tensor->pointer()),
      N, H, W, C, C);
    CHECK_CUDA_ERRORS(cudaGetLastError());

    concat(
      cuda_stream_,
      transformed_rgb_device_, transformed_xyz_map_device_,
      reinterpret_cast<float*>(original_tensor->pointer()),
      N, H, W, C, C);
    CHECK_CUDA_ERRORS(cudaGetLastError());

    // Publish to itself during refinement iterations
    if (iteration_count_ < refine_iterations_-1) {
      auto iterative_rgb_transmitter = iterative_rgb_transmitter_.try_get();
      if (!iterative_rgb_transmitter) {
        GXF_LOG_ERROR("[FoundationposeRender] Failed to get iterative pose array transmitter.");
        return gxf::ToResultCode(iterative_rgb_transmitter);
      }

      auto iterative_point_cloud_transmitter = iterative_point_cloud_transmitter_.try_get();
      if (!iterative_point_cloud_transmitter) {
        GXF_LOG_ERROR("[FoundationposeRender] Failed to get iterative pose array transmitter.");
        return gxf::ToResultCode(iterative_point_cloud_transmitter);
      }

      auto iterative_camera_model_transmitter = iterative_camera_model_transmitter_.try_get();
      if (!iterative_camera_model_transmitter) {
        GXF_LOG_ERROR("[FoundationposeRender] Failed to get iterative pose array transmitter.");
        return gxf::ToResultCode(iterative_camera_model_transmitter);
      }
      iterative_rgb_transmitter.value()->publish(std::move(rgb_message));
      iterative_point_cloud_transmitter.value()->publish(std::move(xyz_message));
      iterative_camera_model_transmitter.value()->publish(std::move(camera_model_message));
    }

    // Update loop and batch counters
    refine_recevied_batches_ += 1;
    if (refine_recevied_batches_ == kNumBatches) {
      refine_recevied_batches_ = 0;
      iteration_count_ += 1;
    }
    if (iteration_count_ == refine_iterations_) {
      iteration_count_ = 0;
    }
  } else {
    // Score mode, and all messages are accumulated.
    // Concat the last sliced output tensors and publish the results
    auto score_output_offset = score_recevied_batches_*N*H*W*2*C;
    concat(
      cuda_stream_,
      reinterpret_cast<float*>(render_rgb_data->basePtr()),
      reinterpret_cast<float*>(render_xyz_map_data->basePtr()),
      score_rendered_output_device_ + score_output_offset,
      N, H, W, C, C);
    CHECK_CUDA_ERRORS(cudaGetLastError());

    concat(
      cuda_stream_,
      transformed_rgb_device_,
      transformed_xyz_map_device_,
      score_original_output_device_ + score_output_offset,
      N, H, W, C, C);
    CHECK_CUDA_ERRORS(cudaGetLastError());

    CHECK_CUDA_ERRORS(cudaMemcpyAsync(
        reinterpret_cast<float*>(rendered_tensor->pointer()), score_rendered_output_device_, rendered_tensor->size(), cudaMemcpyDeviceToDevice, cuda_stream_));
    CHECK_CUDA_ERRORS(cudaMemcpyAsync(
        reinterpret_cast<float*>(original_tensor->pointer()), score_original_output_device_, original_tensor->size(), cudaMemcpyDeviceToDevice, cuda_stream_));
    score_recevied_batches_ = 0;
  }

  CHECK_CUDA_ERRORS(cudaStreamSynchronize(cuda_stream_));
  return gxf::ToResultCode(pose_array_transmitter_->publish(std::move(output_message)));
}

nvidia::gxf::Expected<bool> FoundationposeRender::isAcceptingRequest() {
  return (iteration_count_ == 0 && refine_recevied_batches_ == 0);
}

gxf_result_t FoundationposeRender::stop() noexcept { 
  delete cr_;

  CHECK_CUDA_ERRORS(cudaFree(pose_clip_device_));
  CHECK_CUDA_ERRORS(cudaFree(mesh_faces_device_));
  CHECK_CUDA_ERRORS(cudaFree(rast_out_device_));
  CHECK_CUDA_ERRORS(cudaFree(pts_cam_device_));
  CHECK_CUDA_ERRORS(cudaFree(texcoords_out_device_));
  CHECK_CUDA_ERRORS(cudaFree(color_device_));
  CHECK_CUDA_ERRORS(cudaFree(xyz_map_device_));
  CHECK_CUDA_ERRORS(cudaFree(texcoords_device_));
  CHECK_CUDA_ERRORS(cudaFree(texture_map_device_));

  CHECK_CUDA_ERRORS(cudaFree(transformed_xyz_map_device_));
  CHECK_CUDA_ERRORS(cudaFree(transformed_rgb_device_));
  CHECK_CUDA_ERRORS(cudaFree(score_rendered_output_device_));
  CHECK_CUDA_ERRORS(cudaFree(score_original_output_device_));
  CHECK_CUDA_ERRORS(cudaFree(wp_image_device_));
  CHECK_CUDA_ERRORS(cudaFree(trans_matrix_device_));
  CHECK_CUDA_ERRORS(cudaFree(bbox2d_device_));

  return GXF_SUCCESS;
}
}  // namespace isaac_ros
}  // namespace nvidia