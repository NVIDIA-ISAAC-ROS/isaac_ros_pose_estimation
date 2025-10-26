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

#include "foundationpose_render.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>

#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <cvcuda/OpConvertTo.hpp>
#include <cvcuda/OpFlip.hpp>
#include <cvcuda/OpWarpPerspective.hpp>
#include <opencv2/opencv.hpp>

#include "foundationpose_utils.hpp"
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
constexpr size_t kTexcoordsDim = 2;
constexpr size_t kPTMatrixDim = 3;
constexpr size_t kPoseMatrixLength = 4;
constexpr size_t kNumChannels = 3;
constexpr size_t kOutputRank = 4;
// Number of slices on the input
constexpr int kNumBatches = 6;


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
    GXF_LOG_ERROR("[FoundationposeRender] The transformation matrix is empty! ");
    return GXF_FAILURE;
  }

  if (tfs[0].cols() != tfs[0].rows()) {
    GXF_LOG_ERROR("[FoundationposeRender] The transformation matrix has different rows and cols! ");
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

// Normalize the image to 0-1 using cvcuda, save the result to float_texture_map_tensor
void NormalizeImage(cudaStream_t cuda_stream, uint8_t* texture_map_device,
                    int image_height, int image_width, int image_channels, 
                    nvcv::Tensor& float_texture_map_tensor){
  nvcv::Tensor texture_map_tensor;
  WrapImgPtrToNHWCTensor(texture_map_device, texture_map_tensor, 1, image_height, image_width, image_channels);

  nvcv::TensorShape::ShapeType shape{1, image_height, image_width, image_channels};
  nvcv::TensorShape tensor_shape{shape, "NHWC"};
  float_texture_map_tensor = nvcv::Tensor(tensor_shape, nvcv::TYPE_F32);

  const float scale_factor =  1.0f / 255.0f;
  cvcuda::ConvertTo convert_op;
  convert_op(cuda_stream, texture_map_tensor, float_texture_map_tensor, scale_factor, 0.0f);
  CHECK_CUDA_ERRORS(cudaGetLastError());
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
      debug_, "debug", "Debug", 
      "When enabled, saves intermediate results to disk for debugging", false);
  
  result &= registrar->parameter(
      debug_dir_, "debug_dir", "Debug Directory", 
      "Directory to save intermediate results for debugging", std::string("/tmp/foundationpose"));

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

  result &= registrar->parameter(
      mesh_storage_, "mesh_storage", "Mesh Storage",
      "The mesh storage for mesh reuse");

  return gxf::ToResultCode(result);
}

gxf_result_t FoundationposeRender::start() noexcept {
  GXF_LOG_DEBUG("[FoundationposeRender] FoundationposeRender start");
  // Validate input parameters
  if (refine_iterations_ < 1) {
    GXF_LOG_ERROR("[FoundationposeRender] Refine iterations should be at least 1");
    return GXF_FAILURE;
  }

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
  return GXF_SUCCESS;
}

gxf_result_t FoundationposeRender::AllocateDeviceMemory(
    size_t N, size_t H, size_t W, size_t C, size_t num_vertices, float mesh_diameter, uint32_t total_poses) {

  // Allocate rendering resources if not already cached
  if (!render_data_cached_) {
    size_t rast_out_size = N * H * W * 4 * sizeof(float);
    size_t color_size = N * H * W * C * sizeof(float);
    size_t xyz_map_size = N * H * W * C * sizeof(float);
    size_t texcoords_out_size = N * H * W * kTexcoordsDim * sizeof(float);
    size_t bbox2d_size = N * 4 * sizeof(float);

    CHECK_CUDA_ERRORS(cudaMalloc(&rast_out_device_, rast_out_size));
    CHECK_CUDA_ERRORS(cudaMalloc(&texcoords_out_device_, texcoords_out_size));
    CHECK_CUDA_ERRORS(cudaMalloc(&color_device_, color_size));
    CHECK_CUDA_ERRORS(cudaMalloc(&xyz_map_device_, xyz_map_size));
    CHECK_CUDA_ERRORS(cudaMalloc(&bbox2d_device_, bbox2d_size));
    CHECK_CUDA_ERRORS(cudaMalloc(&transformed_xyz_map_device_, N * H * W * C * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaMalloc(&transformed_rgb_device_, N * H * W * C * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaMalloc(&score_rendered_output_device_, total_poses * 2 * H * W * C * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaMalloc(&score_original_output_device_, total_poses * 2 * H * W * C * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaMalloc(&wp_image_device_, N * H * W * C * sizeof(uint8_t)));
    CHECK_CUDA_ERRORS(cudaMalloc(&trans_matrix_device_, N * 9 * sizeof(float)));

    nvcv::TensorShape::ShapeType shape{N, H, W, C};
    nvcv::TensorShape tensor_shape{shape, "NHWC"};
    render_rgb_tensor_ = nvcv::Tensor(tensor_shape, nvcv::TYPE_F32);
    render_xyz_map_tensor_ = nvcv::Tensor(tensor_shape, nvcv::TYPE_F32);

    cr_ = new CR::CudaRaster();
    render_data_cached_ = true;
  }

  // Handle vertex data separately since it can change with different meshes
  if (num_vertices != num_vertices_cache_) {
    if (pts_cam_device_) {
      CHECK_CUDA_ERRORS(cudaFree(pts_cam_device_));
    }
    if (pose_clip_device_) {
      CHECK_CUDA_ERRORS(cudaFree(pose_clip_device_));
    }
    size_t pts_cam_size = N * num_vertices * kVertexPoints * sizeof(float);
    size_t pose_clip_size = N * num_vertices * 4 * sizeof(float);
    CHECK_CUDA_ERRORS(cudaMalloc(&pts_cam_device_, pts_cam_size));
    CHECK_CUDA_ERRORS(cudaMalloc(&pose_clip_device_, pose_clip_size));
    num_vertices_cache_ = num_vertices;
  }

  return GXF_SUCCESS;
}

gxf_result_t FoundationposeRender::FreeDeviceMemory() {
  if (cr_) {
    delete cr_;
  }

  // Free all CUDA resources
  CHECK_CUDA_ERRORS(cudaFree(pose_clip_device_));
  CHECK_CUDA_ERRORS(cudaFree(rast_out_device_));
  CHECK_CUDA_ERRORS(cudaFree(pts_cam_device_));
  CHECK_CUDA_ERRORS(cudaFree(texcoords_out_device_));
  CHECK_CUDA_ERRORS(cudaFree(color_device_));
  CHECK_CUDA_ERRORS(cudaFree(xyz_map_device_));
  CHECK_CUDA_ERRORS(cudaFree(transformed_xyz_map_device_));
  CHECK_CUDA_ERRORS(cudaFree(transformed_rgb_device_));
  CHECK_CUDA_ERRORS(cudaFree(score_rendered_output_device_));
  CHECK_CUDA_ERRORS(cudaFree(score_original_output_device_));
  CHECK_CUDA_ERRORS(cudaFree(wp_image_device_));
  CHECK_CUDA_ERRORS(cudaFree(trans_matrix_device_));
  CHECK_CUDA_ERRORS(cudaFree(bbox2d_device_));

  return GXF_SUCCESS;
}

void FoundationposeRender::PreparePerspectiveTransformMatrix(
    cudaStream_t cuda_stream, const std::vector<RowMajorMatrix>& tfs,
    float* trans_matrix_device, size_t N) {

  std::vector<float> trans_mat_flattened(N*9, 0);
  for(size_t index = 0; index < N; index++) {
    for (size_t i = 0; i < kPTMatrixDim; i++) {
      for (size_t j = 0; j < kPTMatrixDim; j++) {
        trans_mat_flattened[index*kPTMatrixDim*kPTMatrixDim + i*kPTMatrixDim + j] = tfs[index](i,j);
      }
    }
  }

  CHECK_CUDA_ERRORS(cudaMemcpyAsync(
      trans_matrix_device,
      trans_mat_flattened.data(),
      trans_mat_flattened.size()*sizeof(float),
      cudaMemcpyHostToDevice, 
      cuda_stream));
}

void FoundationposeRender::BatchedWarpPerspective(
    cudaStream_t cuda_stream, const nvcv::Tensor& src_tensor, 
    void* dst_device_ptr, void* trans_matrix_device,
    int num_of_trans_mat, int src_H, int src_W, int dst_H, int dst_W, 
    int C, nvcv::ImageFormat fmt, int interp_flags, const float4& border_value) {

  // Build image batch input
  std::vector<nvcv::Image> wp_src;
  nvcv::ImageDataStridedCuda::Buffer buf_src;
  buf_src.numPlanes           = 1;
  buf_src.planes[0].width     = src_W;
  buf_src.planes[0].height    = src_H;
  buf_src.planes[0].rowStride = src_W*fmt.planePixelStrideBytes(0);
  buf_src.planes[0].basePtr   = reinterpret_cast<NVCVByte*>(src_tensor.exportData<nvcv::TensorDataStridedCuda>()->basePtr());
  auto img_src = nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{fmt, buf_src});

  for (int i = 0; i < num_of_trans_mat; ++i) {
      wp_src.emplace_back(img_src);
  }

  nvcv::ImageBatchVarShape batch_wp_src(num_of_trans_mat);
  batch_wp_src.pushBack(wp_src.begin(), wp_src.end());

  // Build batched output tensor
  std::vector<nvcv::Image> wp_dst;
  for (int i = 0; i < num_of_trans_mat; ++i) {        
      nvcv::ImageDataStridedCuda::Buffer buf;
      buf.numPlanes           = 1;
      buf.planes[0].width     = dst_W;
      buf.planes[0].height    = dst_H;
      buf.planes[0].rowStride = dst_W * fmt.planePixelStrideBytes(0);
      buf.planes[0].basePtr   = reinterpret_cast<NVCVByte*>(dst_device_ptr) + i * dst_H * dst_W * C * fmt.planePixelStrideBytes(0) / C;
      auto img = nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{fmt, buf});
      wp_dst.push_back(img);
  }
  nvcv::ImageBatchVarShape batch_wp_dst(num_of_trans_mat);
  batch_wp_dst.pushBack(wp_dst.begin(), wp_dst.end());

  // Create transformation matrix tensor
  nvcv::TensorDataStridedCuda::Buffer buf_trans_mat;
  buf_trans_mat.strides[1] = sizeof(float);
  buf_trans_mat.strides[0] = 9*buf_trans_mat.strides[1];
  buf_trans_mat.basePtr    = reinterpret_cast<NVCVByte*>(trans_matrix_device);

  auto transMatrixTensor = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
      nvcv::TensorShape({num_of_trans_mat, 9}, nvcv::TENSOR_NW),
      nvcv::TYPE_F32,
      buf_trans_mat
  });

  // Perform the warp perspective operation
  cvcuda::WarpPerspective warpPerspectiveOpBatch(num_of_trans_mat);
  warpPerspectiveOpBatch(cuda_stream, batch_wp_src, batch_wp_dst, transMatrixTensor, interp_flags, NVCV_BORDER_CONSTANT, border_value);
  CHECK_CUDA_ERRORS(cudaGetLastError());
}

gxf_result_t FoundationposeRender::NvdiffrastRender(
    cudaStream_t cuda_stream, 
    std::shared_ptr<const MeshData> mesh_data_ptr, 
    std::vector<Eigen::MatrixXf>& poses, 
    Eigen::Matrix3f& K, 
    Eigen::MatrixXf& bbox2d, 
    int rgb_H, int rgb_W, 
    int H, int W, 
    nvcv::Tensor& flip_color_tensor, 
    nvcv::Tensor& flip_xyz_map_tensor) {
  
  // N represents the number of poses
  size_t N = poses.size();

  // For every pose, transform the vertices to the camera space
  if (kVertexPoints != kPoseMatrixLength - 1) {
    GXF_LOG_ERROR("[FoundationposeRender] The vertice channel should be same as pose matrix length - 1");
    return GXF_FAILURE;
  }
  
  transform_pts(cuda_stream, pts_cam_device_, 
               mesh_data_ptr->mesh_vertices_device, 
               pose_device_, 
               mesh_data_ptr->num_vertices, 
               kVertexPoints, N, kPoseMatrixLength);
  CHECK_CUDA_ERRORS(cudaGetLastError());

  Eigen::Matrix4f projection_mat;
  auto gxf_result = ProjectMatrixFromIntrinsics(projection_mat, K, rgb_H, rgb_W);
  if (gxf_result != GXF_SUCCESS) {
    GXF_LOG_ERROR("[FoundationposeRender] Failed to project the matrix from intrinsics");
    return GXF_FAILURE;
  }

  std::vector<float> h_bbox2d;
  for(int j=0; j<bbox2d.rows(); j++) {
      for(int k=0; k<bbox2d.cols(); k++) {
          h_bbox2d.push_back(bbox2d(j,k));
      }
  }

  CHECK_CUDA_ERRORS(cudaMemcpyAsync(bbox2d_device_, h_bbox2d.data(), N * 4 * sizeof(float), 
                                    cudaMemcpyHostToDevice, cuda_stream));
  generate_pose_clip(cuda_stream, pose_clip_device_, pose_device_, bbox2d_device_, 
                    mesh_data_ptr->mesh_vertices_device, projection_mat, rgb_H, rgb_W, 
                    mesh_data_ptr->num_vertices, N);
  CHECK_CUDA_ERRORS(cudaGetLastError());

  rasterize(
      cuda_stream, cr_,
      pose_clip_device_, mesh_data_ptr->mesh_faces_device, rast_out_device_,
      mesh_data_ptr->num_vertices, mesh_data_ptr->num_faces,
      H, W, N);
  CHECK_CUDA_ERRORS(cudaGetLastError());

  interpolate(
      cuda_stream,
      pts_cam_device_, rast_out_device_, mesh_data_ptr->mesh_faces_device, xyz_map_device_,
      mesh_data_ptr->num_vertices, mesh_data_ptr->num_faces, kVertexPoints,
      H, W, N);
  CHECK_CUDA_ERRORS(cudaGetLastError());

  if (mesh_data_ptr->has_tex) {
    interpolate(
        cuda_stream,
        mesh_data_ptr->texcoords_device, rast_out_device_, mesh_data_ptr->mesh_faces_device, texcoords_out_device_,
        mesh_data_ptr->num_vertices, mesh_data_ptr->num_faces, kTexcoordsDim,
        H, W, N);
    CHECK_CUDA_ERRORS(cudaGetLastError());

    auto float_texture_map_data = float_texture_map_tensor_.exportData<nvcv::TensorDataStridedCuda>();
    texture(
        cuda_stream,
        reinterpret_cast<float*>(float_texture_map_data->basePtr()),
        texcoords_out_device_,
        color_device_,
        mesh_data_ptr->texture_map_height, mesh_data_ptr->texture_map_width, mesh_data_ptr->texture_map_channels,
        1, H, W, N);
    CHECK_CUDA_ERRORS(cudaGetLastError());
  } else {
    auto float_texture_map_data = float_texture_map_tensor_.exportData<nvcv::TensorDataStridedCuda>();
    // The texture map is a list of vertex colors, needs broadcasting in interpolate to avoid illegal memory access
    interpolate(
        cuda_stream,
        reinterpret_cast<float*>(float_texture_map_data->basePtr()), rast_out_device_, mesh_data_ptr->mesh_faces_device, color_device_,
        mesh_data_ptr->num_vertices, mesh_data_ptr->num_faces, kVertexPoints,
        H, W, N, 1);
    CHECK_CUDA_ERRORS(cudaGetLastError());
  }

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

void FoundationposeRender::SaveRGBImage(
    cudaStream_t stream, float* rgb_data, size_t N, 
    size_t H, size_t W, size_t C, const gxf::Timestamp* timestamp,
    const std::string& image_type) {

  // Create debug directory structure with proper subfolder
  std::string data_type = image_type + "_rgb";
  std::string debug_dir = CreateDebugDirectory(timestamp, data_type);

  if (C != kNumChannels) {
    GXF_LOG_WARNING(
        "[FoundationposeRender] RGB image has %zu channels, expected %zu. "
        "Color visualization may be incorrect.", 
        C, kNumChannels);
  }

  std::vector<float> rgb_host(N * H * W * C);

  // Copy data from GPU to CPU
  CHECK_CUDA_ERRORS(cudaMemcpyAsync(
    rgb_host.data(), rgb_data, N * H * W * C * sizeof(float), 
    cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA_ERRORS(cudaStreamSynchronize(stream));

  // Save each image in the batch
  for (size_t i = 0; i < N; i++) {
    cv::Mat rgb_image(H, W, CV_32FC3);

    for (size_t h = 0; h < H; h++) {
      for (size_t w = 0; w < W; w++) {
        size_t idx = (i * H * W + h * W + w) * C;

        // OpenCV stores colors in BGR order, rather than RGB.
        rgb_image.at<cv::Vec3f>(h, w)[0] = rgb_host[idx + 2]; 
        rgb_image.at<cv::Vec3f>(h, w)[1] = rgb_host[idx + 1];
        rgb_image.at<cv::Vec3f>(h, w)[2] = rgb_host[idx];
      }
    }

    cv::Mat rgb_image_8bit;
    rgb_image.convertTo(rgb_image_8bit, CV_8UC3, 255.0);

    std::stringstream filename;
    int batch_idx = (mode_.get() == "refine") ? refine_received_batches_ : score_received_batches_;
    filename << debug_dir << "/batch" << batch_idx << "_pose" << i << ".png";

    cv::imwrite(filename.str(), rgb_image_8bit);
    GXF_LOG_DEBUG("[FoundationposeRender] Saved %s RGB image to %s", image_type.c_str(), filename.str().c_str());
  }

  GXF_LOG_DEBUG("[FoundationposeRender] Saved %zu %s RGB images to %s", N, image_type.c_str(), debug_dir.c_str());
}

void FoundationposeRender::SaveXYZImage(
    cudaStream_t stream, float* xyz_data, size_t N, 
    size_t H, size_t W, size_t C, const gxf::Timestamp* timestamp,
    const std::string& image_type) {
  
  // Create debug directory structure with proper subfolder
  std::string data_type = image_type + "_xyz";
  std::string debug_dir = CreateDebugDirectory(timestamp, data_type);
  
  if (C != 3) {
    GXF_LOG_WARNING(
        "[FoundationposeRender] XYZ image has %zu channels, expected %zu. "
        "Point cloud visualization may be incorrect.", 
        C, kNumChannels);
  }

  std::vector<float> xyz_host(N * H * W * C);
  
  CHECK_CUDA_ERRORS(cudaMemcpyAsync(
    xyz_host.data(), xyz_data, N * H * W * C * sizeof(float), 
    cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA_ERRORS(cudaStreamSynchronize(stream));
  
  // Save each point cloud image in the batch
  for (size_t i = 0; i < N; i++) {
    cv::Mat xyz_color(H, W, CV_8UC3, cv::Scalar(0, 0, 0));

    // Find min/max values for normalization
    float x_min = std::numeric_limits<float>::max();
    float x_max = std::numeric_limits<float>::lowest();
    float y_min = std::numeric_limits<float>::max();
    float y_max = std::numeric_limits<float>::lowest();
    float z_min = std::numeric_limits<float>::max();
    float z_max = std::numeric_limits<float>::lowest();

    // First pass: find min/max values
    for (size_t h = 0; h < H; h++) {
      for (size_t w = 0; w < W; w++) {
        size_t idx = (i * H * W + h * W + w) * C;
        float x = xyz_host[idx];
        float y = xyz_host[idx + 1];
        float z = xyz_host[idx + 2];
        
        if (x != 0) {
          x_min = std::min(x_min, x);
          x_max = std::max(x_max, x);
        }
        if (y != 0) {
          y_min = std::min(y_min, y);
          y_max = std::max(y_max, y);
        }
        if (z != 0) {
          z_min = std::min(z_min, z);
          z_max = std::max(z_max, z);
        }
      }
    }

    // Second pass: create false-color visualization
    for (size_t h = 0; h < H; h++) {
      for (size_t w = 0; w < W; w++) {
        size_t idx = (i * H * W + h * W + w) * C;
        float x = xyz_host[idx];
        float y = xyz_host[idx + 1];
        float z = xyz_host[idx + 2];

        if (x == 0 && y == 0 && z == 0)
          continue;

        uint8_t r = (x_max > x_min) ? static_cast<uint8_t>(255 * (x - x_min) / (x_max - x_min)) : 0;
        uint8_t g = (y_max > y_min) ? static_cast<uint8_t>(255 * (y - y_min) / (y_max - y_min)) : 0;
        uint8_t b = (z_max > z_min) ? static_cast<uint8_t>(255 * (z - z_min) / (z_max - z_min)) : 0;

        // OpenCV stores colors in BGR order, rather than RGB.
        xyz_color.at<cv::Vec3b>(h, w) = cv::Vec3b(b, g, r);
      }
    }

    std::stringstream xyz_filename;
    int batch_idx = (mode_.get() == "refine") ? refine_received_batches_ : score_received_batches_;
    xyz_filename << debug_dir << "/batch" << batch_idx << "_pose" << i << ".png";

    cv::imwrite(xyz_filename.str(), xyz_color);
    GXF_LOG_DEBUG("[FoundationposeRender] Saved %s XYZ image to %s", image_type.c_str(), xyz_filename.str().c_str());
  }
  
  GXF_LOG_INFO("[FoundationposeRender] Saved %zu %s point cloud images to %s", N, image_type.c_str(), debug_dir.c_str());
}

// Create directory structure for debug images and point clouds
std::string FoundationposeRender::CreateDebugDirectory(const gxf::Timestamp* timestamp, const std::string& data_type) {
  // Get timestamp to use for the directory
  int64_t ts = 0;
  if (timestamp != nullptr) {
    // Use the message timestamp in nanoseconds
    ts = timestamp->acqtime;
  } else {
    // Fallback to current time if timestamp is not available
    auto time_now = std::chrono::system_clock::now();
    ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
      time_now.time_since_epoch()).count();
  }
  
  // Create common base directory path
  std::stringstream ss;
  ss << debug_dir_.get() << "/fp_debug_" << ts;
  std::string base_dir = ss.str();
  
  // Create base directory if it doesn't exist
  if (current_save_dir_ != base_dir) {
    if (!std::filesystem::exists(base_dir)) {
      std::filesystem::create_directories(base_dir);
    }
    current_save_dir_ = base_dir;
  }
  
  // Create mode-specific subdirectory
  std::string mode_dir = current_save_dir_ + "/" + mode_.get();
  if (!std::filesystem::exists(mode_dir)) {
    std::filesystem::create_directories(mode_dir);
  }
  
  // Create iteration subdirectory if in refine mode
  std::string iter_dir = mode_dir;
  if (mode_.get() == "refine" && refine_iterations_ > 1) {
    iter_dir = mode_dir + "/iteration_" + std::to_string(iteration_count_);
    if (!std::filesystem::exists(iter_dir)) {
      std::filesystem::create_directories(iter_dir);
    }
  }

  // Create data type specific subfolder
  std::string data_dir = iter_dir + "/" + data_type;
  if (!std::filesystem::exists(data_dir)) {
    std::filesystem::create_directories(data_dir);
  }

  return data_dir;
}

gxf_result_t FoundationposeRender::tick() noexcept {
  GXF_LOG_DEBUG("[FoundationposeRender] tick");

  // Get mesh data from sampling component if not already set
  auto mesh_data_ptr = mesh_storage_.get()->GetMeshData();

  // For pure color texture (vertex colors), we can reuse the cached version if the size matches
  if (float_texture_map_tensor_.empty() || texture_path_cache_ != mesh_data_ptr->texture_path
      || mesh_data_ptr->texture_map_height != float_texture_map_tensor_.shape()[1]
      || mesh_data_ptr->texture_map_width != float_texture_map_tensor_.shape()[2]
      || mesh_data_ptr->texture_map_channels != float_texture_map_tensor_.shape()[3]) {
    NormalizeImage(
      cuda_stream_,
      mesh_data_ptr->texture_map_device, 
      mesh_data_ptr->texture_map_height, 
      mesh_data_ptr->texture_map_width, 
      mesh_data_ptr->texture_map_channels, 
      float_texture_map_tensor_);
    texture_path_cache_ = mesh_data_ptr->texture_path;
  }

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

  const size_t N = poses_handle->shape().dimension(0);
  const size_t pose_rows = poses_handle->shape().dimension(1);
  const size_t pose_cols = poses_handle->shape().dimension(2);

  if (N == 0) {
    GXF_LOG_ERROR("[FoundationposeRender] The received pose is empty");
    return GXF_FAILURE;
  }

  // Each pose should be a 4*4 matrix
  if (pose_rows != kPoseMatrixLength || pose_cols != kPoseMatrixLength) {
    GXF_LOG_ERROR("[FoundationposeRender] The received pose has wrong dimension");
    return GXF_FAILURE;
  }

  if (poses_handle->size() != N * pose_rows * pose_cols * sizeof(float)) {
    GXF_LOG_ERROR("[FoundationposeRender] Unexpected size of the pose tensor");
    return GXF_FAILURE;
  }

  const uint32_t rgb_H = rgb_img_info.height;
  const uint32_t rgb_W = rgb_img_info.width;

  const uint32_t W = resized_image_width_;
  const uint32_t H = resized_image_height_;
  const uint32_t C = kNumChannels;

  const uint32_t total_poses = N * kNumBatches;

  // Create Eigen matrix from the gxf camera model message
  Eigen::Matrix3f K;
  K << gxf_camera_model->focal_length.x, 0.0, gxf_camera_model->principal_point.x,
       0.0, gxf_camera_model->focal_length.y, gxf_camera_model->principal_point.y,
       0.0, 0.0, 1.0;

  // Allocate device memory
  auto alloc_result = AllocateDeviceMemory(N, H, W, C, mesh_data_ptr->num_vertices, 
                                        mesh_data_ptr->mesh_diameter, total_poses);
  if (alloc_result != GXF_SUCCESS) {
    GXF_LOG_ERROR("[FoundationposeRender] Failed to allocate device memory");
    return alloc_result;
  }

  std::vector<float> pose_host(N * pose_rows * pose_cols);
  pose_device_ = reinterpret_cast<float*>(poses_handle->pointer());
  CHECK_CUDA_ERRORS(cudaMemcpyAsync(
      pose_host.data(), poses_handle->pointer(), poses_handle->size(), cudaMemcpyDeviceToHost, cuda_stream_));
  cudaStreamSynchronize(cuda_stream_);

  // Construct pose matrix from pointer
  std::vector<Eigen::MatrixXf> poses;
  for (size_t i = 0; i < N; i++) {
    Eigen::Map<Eigen::MatrixXf> mat(
        pose_host.data() + i * pose_rows * pose_cols, pose_rows, pose_cols);
    poses.push_back(mat);
  }
  
  Eigen::Vector2i out_size = {H, W};
  auto tfs = ComputeCropWindowTF(poses, K, out_size, crop_ratio_, mesh_data_ptr->mesh_diameter);
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

  // Render the object using give poses, passing member variables explicitly
  gxf_result = NvdiffrastRender(
      cuda_stream_, mesh_data_ptr, poses, K, bbox2d, rgb_H, rgb_W, H, W, 
      render_rgb_tensor_, render_xyz_map_tensor_);
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

  const float scale_factor = 1.0f / 255.0f;
  int num_of_trans_mat = N;
  cvcuda::ConvertTo convert_op;

  // Prepare transformation matrices
  PreparePerspectiveTransformMatrix(cuda_stream_, tfs, trans_matrix_device_, N);

  // Process RGB image and XYZ map using BatchedWarpPerspective
  const nvcv::ImageFormat fmt_rgb = nvcv::FMT_RGB8;
  BatchedWarpPerspective(
      cuda_stream_, 
      rgb_tensor, 
      wp_image_device_, 
      trans_matrix_device_, 
      num_of_trans_mat,
      rgb_H, rgb_W, 
      H, W, 
      C,
      fmt_rgb,
      rgb_flags,
      border_value);

  const nvcv::ImageFormat fmt_xyz = nvcv::FMT_RGBf32;
  BatchedWarpPerspective(
      cuda_stream_, 
      xyz_map_tensor, 
      transformed_xyz_map_device_, 
      trans_matrix_device_, 
      num_of_trans_mat,
      rgb_H, rgb_W, 
      H, W, 
      C,
      fmt_xyz,
      xyz_flags,
      border_value);

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
      N, W * H, mesh_data_ptr->mesh_diameter / 2, min_depth_, max_depth_);
  CHECK_CUDA_ERRORS(cudaGetLastError());

  auto render_rgb_data = render_rgb_tensor_.exportData<nvcv::TensorDataStridedCuda>();
  auto render_xyz_map_data = render_xyz_map_tensor_.exportData<nvcv::TensorDataStridedCuda>();

  threshold_and_downscale_pointcloud(
      cuda_stream_,
      reinterpret_cast<float*>(render_xyz_map_data->basePtr()),
      reinterpret_cast<float*>(poses_handle->pointer()),
      N, W * H, mesh_data_ptr->mesh_diameter / 2, min_depth_, max_depth_);
  CHECK_CUDA_ERRORS(cudaGetLastError());

  // Score mode, accumulation stage
  // Only accumulate the output tensors and return
  if (mode_.get() == "score" && score_received_batches_ < kNumBatches - 1) {
    auto score_output_offset = score_received_batches_*N*H*W*2*C;
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

    auto maybe_timestamp_final = rgb_message.get<gxf::Timestamp>();
    const gxf::Timestamp* timestamp_ptr_final = nullptr;
    if (maybe_timestamp_final) {
      timestamp_ptr_final = maybe_timestamp_final.value();
    }

    GXF_LOG_DEBUG("[FoundationposeRender] Saving debug images in score mode (final)");
    if (debug_.get()) {
      SaveRGBImage(
          cuda_stream_,
          reinterpret_cast<float*>(render_rgb_data->basePtr()),
          N, H, W, C,
          timestamp_ptr_final, "rendered");
      
      SaveXYZImage(
          cuda_stream_,
          reinterpret_cast<float*>(render_xyz_map_data->basePtr()),
          N, H, W, C,
          timestamp_ptr_final, "rendered");
          
      SaveRGBImage(
          cuda_stream_,
          transformed_rgb_device_,
          N, H, W, C,
          timestamp_ptr_final, "original");
          
      SaveXYZImage(
          cuda_stream_,
          transformed_xyz_map_device_,
          N, H, W, C,
          timestamp_ptr_final, "original");
    }
    score_received_batches_ += 1;
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

  // Prepare output GXF tensor
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
    
    // Save rendered images in refine mode
    auto maybe_timestamp_refine = rgb_message.get<gxf::Timestamp>();
    const gxf::Timestamp* timestamp_ptr_refine = nullptr;
    if (maybe_timestamp_refine) {
      timestamp_ptr_refine = maybe_timestamp_refine.value();
    }
    
    if (debug_.get()) {
      GXF_LOG_DEBUG("[FoundationposeRender] Saving debug images in refine mode");
      
      // Save rendered RGB and point cloud images
      SaveRGBImage(
          cuda_stream_,
          reinterpret_cast<float*>(render_rgb_data->basePtr()),
          N, H, W, C, 
          timestamp_ptr_refine, "rendered");
      
      SaveXYZImage(
          cuda_stream_,
          reinterpret_cast<float*>(render_xyz_map_data->basePtr()),
          N, H, W, C,
          timestamp_ptr_refine, "rendered");
          
      // Save original (transformed) RGB and point cloud images
      SaveRGBImage(
          cuda_stream_,
          transformed_rgb_device_,
          N, H, W, C,
          timestamp_ptr_refine, "original");
          
      SaveXYZImage(
          cuda_stream_,
          transformed_xyz_map_device_,
          N, H, W, C,
          timestamp_ptr_refine, "original");
    }

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
    refine_received_batches_ += 1;
    if (refine_received_batches_ == kNumBatches) {
      refine_received_batches_ = 0;
      iteration_count_ += 1;
    }
    if (iteration_count_ == refine_iterations_) {
      iteration_count_ = 0;
    }
  } else {
    // Score mode, and all messages are accumulated.
    // Concat the last sliced output tensors and publish the results
    auto score_output_offset = score_received_batches_*N*H*W*2*C;
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

    auto maybe_timestamp_final = rgb_message.get<gxf::Timestamp>();
    const gxf::Timestamp* timestamp_ptr_final = nullptr;
    if (maybe_timestamp_final) {
      timestamp_ptr_final = maybe_timestamp_final.value();
    }

    GXF_LOG_DEBUG("[FoundationposeRender] Saving debug images in score mode (final)");
    if (debug_.get()) {
      SaveRGBImage(
          cuda_stream_,
          reinterpret_cast<float*>(render_rgb_data->basePtr()),
          N, H, W, C,
          timestamp_ptr_final, "rendered");
      
      SaveXYZImage(
          cuda_stream_,
          reinterpret_cast<float*>(render_xyz_map_data->basePtr()),
          N, H, W, C,
          timestamp_ptr_final, "rendered");
          
      SaveRGBImage(
          cuda_stream_,
          transformed_rgb_device_,
          N, H, W, C,
          timestamp_ptr_final, "original");
          
      SaveXYZImage(
          cuda_stream_,
          transformed_xyz_map_device_,
          N, H, W, C,
          timestamp_ptr_final, "original");
    }
    score_received_batches_ = 0;
  }

  CHECK_CUDA_ERRORS(cudaStreamSynchronize(cuda_stream_));
  return gxf::ToResultCode(pose_array_transmitter_->publish(std::move(output_message)));
}

gxf_result_t FoundationposeRender::stop() noexcept { 
  return FreeDeviceMemory();
}

}  // namespace isaac_ros
}  // namespace nvidia