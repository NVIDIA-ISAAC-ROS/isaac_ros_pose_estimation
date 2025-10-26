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

#include "foundationpose_render.cu.hpp"

#include <iostream>

void RasterizeCudaFwdShaderKernel(const RasterizeCudaFwdShaderParams p);
void InterpolateFwdKernel(const InterpolateKernelParams p);
void TextureFwdKernelLinear1(const TextureKernelParams p);

namespace nvidia {
namespace isaac_ros {

__device__ float clamp_func(float f, float a, float b) {
  return fmaxf(a, fminf(f, b));
}

__global__ void clamp_kernel(float* input, float min_value, float max_value, int N) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  // Check the boundaries
  if (idx >= N) {
    return;
  }
  input[idx] = clamp_func(input[idx], min_value, max_value);
}


/*
This kernel performs:
 1. thresholdingof the point cloud
 2. subtraction of the position of pose array from the pointcloud
 3. downscaling of the point cloud
 
 pose_array_input is of size N*16, where N is the number of poses. 16  = transformation_mat_size
 pointcloud_input is of size N*n_points*3, where N is the number of poses
    and n_points is the number of points in the point cloud.
 
 It subtracts the pose transformation from each point in the cloud,
 1. checks if the z-component of the point is below "min_depth" and sets it to zero if it is
 2. and applies a downscaling factor to reduce the number of points.
 3. Then it checks if the absolute value of any of the x, y, or z components of the point
    is above "max_depth" and sets it to zero if it is.

 The result is stored back in the input array.
*/
__global__ void threshold_and_downscale_pointcloud_kernel(
    float* input, float* pose_array_input, int N, int n_points, float downscale_factor,
    float min_depth, float max_depth) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= N * n_points) {
    return;  // Check the boundaries
  }

  int pose_idx = idx / n_points;

  // 16 is the size of pose transformation matrix
  float pose_x = pose_array_input[16 * pose_idx + 12];
  float pose_y = pose_array_input[16 * pose_idx + 13];
  float pose_z = pose_array_input[16 * pose_idx + 14];

  // Calculate the index for the x, y, and z components of the point
  int x_idx = idx * 3;
  int y_idx = x_idx + 1;
  int z_idx = x_idx + 2;

  bool invalid_flag = false;
  // Any points with z below min_depth is set to 0
  if (input[z_idx] < min_depth) {
    invalid_flag = true;
  }

  input[x_idx] -= pose_x;
  input[y_idx] -= pose_y;
  input[z_idx] -= pose_z;

  // Divide all values by downscale_factor
  input[x_idx] /= downscale_factor;
  input[y_idx] /= downscale_factor;
  input[z_idx] /= downscale_factor;

  // Any points with absolute value(x,y or z) above max_depth is set to 0
  if (fabs(input[x_idx]) > max_depth || invalid_flag) {
    input[x_idx] = 0.0f;
  }
  if (fabs(input[y_idx]) > max_depth || invalid_flag) {
    input[y_idx] = 0.0f;
  }

  if (fabs(input[z_idx]) > max_depth || invalid_flag) {
    input[z_idx] = 0.0f;
  }
  return;
}


// concat two NHWC array on the last dimension
__global__ void concat_kernel(
    float* input_a, float* input_b, float* output, int N, int H, int W, int C1, int C2) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  // Check the boundaries
  if (idx >= N * H * W) {
    return;
  }

  for (int i = 0; i < C1; i++) {
    output[idx * (C1 + C2) + i] = input_a[idx * C1 + i];
  }

  for (int i = 0; i < C2; i++) {
    output[idx * (C1 + C2) + C1 + i] = input_b[idx * C2 + i];
  }
}

// Transform points using transformation matrices
// Each thread will transform a point using N transformation matrices and save to output
__global__ void transform_pts_kernel(
  float* output, const float* pts, const float* tfs, int pts_num, int pts_channel, int tfs_num, int tfs_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < pts_num) {
    for (int i = 0; i < tfs_num; i++) {
      float* transformed_matrix = output + i * pts_num * (tfs_dim - 1) + idx * (tfs_dim - 1);
      const float* submatrix = tfs + i * tfs_dim * tfs_dim;
      const float* last_col = tfs + i * tfs_dim * tfs_dim + (tfs_dim - 1) * tfs_dim;

      for (int j = 0; j < pts_channel; j++) {
        float new_row = 0.0f;
        for (int k = 0; k < tfs_dim - 1; k++) {
          new_row += submatrix[k * tfs_dim + j] * pts[idx * pts_channel + k];
        }
        new_row += last_col[j];
        transformed_matrix[j] = new_row;
      }
    }
  }
}

__device__ float dot(const float4& v1, const float4& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
}


// From OpenCV camera (cvcam) coordinate system to the OpenGL camera (glcam) coordinate system
const Eigen::Matrix4f kGLCamInCVCam =
    (Eigen::Matrix4f(4, 4) << 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1).finished();
__constant__ float d_projection_fused[16];

// Function to copy data to constant memory
void copy_to_constant_memory(const Eigen::Matrix4f& projection_fused) {
    std::vector<float> flattened_projection_fused;
    for(int i=0;i<projection_fused.rows();i++){
        for(int j=0;j<projection_fused.cols();j++){
            flattened_projection_fused.push_back(projection_fused(i,j));
        }
    }

    cudaMemcpyToSymbol(d_projection_fused,
                       flattened_projection_fused.data(),
                       flattened_projection_fused.size() * sizeof(float));
}

// Multiply a 4x4 matrix with a 4x1 vector
__device__ float4 matrix_vec_mul4x4(const float4* mat, const float4& vec) {
    return make_float4(
        dot(mat[0], vec),
        dot(mat[1], vec),
        dot(mat[2], vec),
        dot(mat[3], vec)
    );
}

// Transform 3D points using the pose matrix and the bounding box transformation matrix
__global__ void pose_clip_kernel_fused(
    float* output, const float* d_poses, const float* d_bbox2d, 
    const float* d_pts, const int n_poses, const int n_pts, 
    const int rgb_H, const int rgb_W) {

    __shared__ float shared_mem[32];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pts) { return; }

    // Load point using float3 
    // Convert float3 to float4 by adding a homogeneous coordinate w=1.0f
    float3 pt3 = reinterpret_cast<const float3*>(d_pts)[idx];
    float4 pt = make_float4(pt3.x, pt3.y, pt3.z, 1.0f);
    
    // Shared memory for storing a single pose mat and bbox transformation matrix
    size_t stride = sizeof(float4);
    float4* shared_tf = reinterpret_cast<float4*>(shared_mem);
    float4* shared_bbox = reinterpret_cast<float4*>(shared_mem + stride);

    for (int i = 0; i < n_poses; i++) {
        float4* transformed_matrix = reinterpret_cast<float4*>(output + i * n_pts * 4 + idx * 4);
        // Compute bbox transformation (only done by first 4 threads per block)
        if (threadIdx.x == 0) {        
            const float4* pose = reinterpret_cast<const float4*>(d_poses + i * stride);
            // Load the projection matrix into shared memory (vectorized)
            shared_tf[0] = make_float4(pose[0].x,pose[1].x,pose[2].x,pose[3].x);
            shared_tf[1] = make_float4(pose[0].y,pose[1].y,pose[2].y,pose[3].y);
            shared_tf[2] = make_float4(pose[0].z,pose[1].z,pose[2].z,pose[3].z);
            shared_tf[3] = make_float4(pose[0].w,pose[1].w,pose[2].w,pose[3].w);
            float l = d_bbox2d[i * 4];
            float t = rgb_H - d_bbox2d[i * 4 + 1];
            float r = d_bbox2d[i * 4 + 2];
            float b = rgb_H - d_bbox2d[i * 4 + 3];

            shared_bbox[0] = make_float4(rgb_W / (r - l), 0, 0, (rgb_W - r - l) / (r - l));
            shared_bbox[1] = make_float4(0, rgb_H / (t - b), 0, (rgb_H - t - b) / (t - b));
            shared_bbox[2] = make_float4(0, 0, 1, 0);
            shared_bbox[3] = make_float4(0, 0, 0, 1);
        }

        __syncthreads();

        // Perform transformation using vectorized operations
        float4 result = matrix_vec_mul4x4(shared_tf, pt);
        result = matrix_vec_mul4x4(reinterpret_cast<const float4*>(d_projection_fused), result);
        result = matrix_vec_mul4x4(shared_bbox, result);

        *transformed_matrix = result;

        // Make sure all threads have read the shared memory before thread 0 writes to it in the next iteration
        __syncthreads();
    }
}

void clamp(cudaStream_t stream, float* input, float min_value, float max_value, int N) {
  int block_size = 256;
  int grid_size = (N + block_size - 1) / block_size;

  clamp_kernel<<<grid_size, block_size, 0, stream>>>(input, min_value, max_value, N);
}

void generate_pose_clip(
    cudaStream_t stream, float* d_pose_clip, const float* d_pose, const float* d_bbox2d, const float* d_mesh_vertices,
    const Eigen::Matrix4f& projection_mat, int rgb_H, int rgb_W, int n_pts, int N) {

    copy_to_constant_memory(projection_mat*kGLCamInCVCam);
    // Define block and grid size
    int blockSize = 256;
    int gridSize = (n_pts + blockSize - 1) / blockSize;

    pose_clip_kernel_fused<<<gridSize, blockSize, 0, stream>>>(
        d_pose_clip, d_pose, d_bbox2d, d_mesh_vertices, N, n_pts, rgb_H, rgb_W);
}

void threshold_and_downscale_pointcloud(
    cudaStream_t stream, float* pointcloud_input, float* pose_array_input, int N, int n_points, float downscale_factor,
    float min_depth, float max_depth) {
  // Launch n_points threads
  int block_size = 256;
  int grid_size = ((N * n_points) + block_size - 1) / block_size;

  threshold_and_downscale_pointcloud_kernel<<<grid_size, block_size, 0, stream>>>(
      pointcloud_input, pose_array_input, N, n_points, downscale_factor, min_depth, max_depth);
}

void transform_pts(
  cudaStream_t stream, float* output, const float* pts, const float* tfs, int pts_num, int pts_channel, int tfs_num, int tfs_dim) {
  // Lannch pts_num threads, each thread handle a point and N transformation matrices
  int block_size = 256;
  int grid_size = (pts_num + block_size - 1) / block_size;
  transform_pts_kernel<<<grid_size, block_size, 0, stream>>>(output, pts, tfs, pts_num, pts_channel, tfs_num, tfs_dim);
}

void concat(
  cudaStream_t stream, float* input_a, float* input_b, float* output, int N, int H, int W, int C1, int C2) {
  // Launch N*H*W threads, each thread handle a vector of size C
  int block_size = 256;
  int grid_size = (N * H * W + block_size - 1) / block_size;

  concat_kernel<<<grid_size, block_size, 0, stream>>>(input_a, input_b, output, N, H, W, C1, C2);
}

void rasterize(
    cudaStream_t stream, CR::CudaRaster* cr, float* pos_ptr, int32_t* tri_ptr, float* out, int pos_count, int tri_count,
    int H, int W, int C) {
  const int32_t* range_ptr = 0;

  bool enablePeel = false;
  cr->setViewportSize(W, H, C);
  cr->setVertexBuffer((void*)pos_ptr, pos_count);
  cr->setIndexBuffer((void*)tri_ptr, tri_count);
  cr->setRenderModeFlags(0);

  cr->deferredClear(0u);
  bool success = cr->drawTriangles(range_ptr, enablePeel, stream);

  // Populate pixel shader kernel parameters.
  RasterizeCudaFwdShaderParams p;
  p.pos = pos_ptr;
  p.tri = tri_ptr;
  p.in_idx = (const int*)cr->getColorBuffer();
  p.out = out;
  p.numTriangles = tri_count;
  p.numVertices = pos_count;
  p.width = W;
  p.height = H;
  p.depth = C;

  p.instance_mode = 1;
  p.xs = 2.f / (float)p.width;
  p.xo = 1.f / (float)p.width - 1.f;
  p.ys = 2.f / (float)p.height;
  p.yo = 1.f / (float)p.height - 1.f;

  // Choose launch parameters.
  dim3 blockSize = getLaunchBlockSize(
      RAST_CUDA_FWD_SHADER_KERNEL_BLOCK_WIDTH, RAST_CUDA_FWD_SHADER_KERNEL_BLOCK_HEIGHT, p.width,
      p.height);
  dim3 gridSize = getLaunchGridSize(blockSize, p.width, p.height, p.depth);

  // Launch CUDA kernel.
  void* args[] = {&p};
  CHECK_CUDA_ERRORS(
      cudaLaunchKernel((void*)RasterizeCudaFwdShaderKernel, gridSize, blockSize, args, 0, stream));
}

void interpolate(
    cudaStream_t stream, float* attr_ptr, float* rast_ptr, int32_t* tri_ptr, float* out, int num_vertices,
    int num_triangles, int attr_dim, int H, int W, int C, int attr_bc) {
  int instance_mode = attr_dim > 2 ? 1 : 0;

  InterpolateKernelParams p = {};  // Initialize all fields to zero.
  p.instance_mode = instance_mode;
  p.numVertices = num_vertices;
  p.numAttr = attr_dim;
  p.numTriangles = num_triangles;
  p.height = H;
  p.width = W;
  p.depth = C;

  // Get input pointers.
  p.attr = attr_ptr;
  p.rast = rast_ptr;
  p.tri = tri_ptr;
  p.attrBC = attr_bc;
  p.out = out;

  // Choose launch parameters.
  dim3 blockSize = getLaunchBlockSize(
      IP_FWD_MAX_KERNEL_BLOCK_WIDTH, IP_FWD_MAX_KERNEL_BLOCK_HEIGHT, p.width, p.height);
  dim3 gridSize = getLaunchGridSize(blockSize, p.width, p.height, p.depth);

  // Launch CUDA kernel.
  void* args[] = {&p};
  void* func = (void*)InterpolateFwdKernel;
  CHECK_CUDA_ERRORS(cudaLaunchKernel(func, gridSize, blockSize, args, 0, stream));
}

void texture(
    cudaStream_t stream, float* tex_ptr, float* uv_ptr, float* out, int tex_height, int tex_width, int tex_channel,
    int tex_depth, int H, int W, int N) {
  TextureKernelParams p = {};  // Initialize all fields to zero.
  p.enableMip = false;
  p.filterMode = TEX_MODE_LINEAR;
  p.boundaryMode = TEX_BOUNDARY_MODE_WRAP;

  p.texDepth = tex_depth;
  p.texHeight = tex_height;
  p.texWidth = tex_width;
  p.channels = tex_channel;

  p.n = N;
  p.imgHeight = H;
  p.imgWidth = W;

  // Get input pointers.
  p.tex[0] = tex_ptr;
  p.uv = uv_ptr;
  p.mipLevelBias = NULL;

  p.out = out;

  // Choose kernel variants based on channel count.
  void* args[] = {&p};

  // Choose launch parameters for texture lookup kernel.
  dim3 blockSize = getLaunchBlockSize(
      TEX_FWD_MAX_KERNEL_BLOCK_WIDTH, TEX_FWD_MAX_KERNEL_BLOCK_HEIGHT, p.imgWidth, p.imgHeight);
  dim3 gridSize = getLaunchGridSize(blockSize, p.imgWidth, p.imgHeight, p.n);

  void* func = (void*)TextureFwdKernelLinear1;
  CHECK_CUDA_ERRORS(cudaLaunchKernel(func, gridSize, blockSize, args, 0, stream));
}

}  // namespace isaac_ros
}  // namespace nvidia
