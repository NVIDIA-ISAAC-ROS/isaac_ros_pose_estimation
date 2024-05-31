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

#ifndef NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_RENDER_HPP_
#define NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_RENDER_HPP_

#include <algorithm>
#include <Eigen/Dense>

#include <nvcv/Tensor.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "gxf/core/entity.hpp"
#include "gxf/core/gxf.h"
#include "gxf/core/parameter.hpp"
#include "gxf/core/parameter_parser_std.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/memory_buffer.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

#include "foundationpose_render.cu.hpp"

namespace nvidia {
namespace isaac_ros {

// GXF codelet takes the pose estimations and render them into image and point clouds.
// The rendered results are then concatenated and sent to refine or score network for further processing
class FoundationposeRender : public gxf::Codelet {
 public:
  gxf_result_t start() noexcept override;
  gxf_result_t tick() noexcept override;
  gxf_result_t stop() noexcept override;
  gxf_result_t registerInterface(gxf::Registrar* registrar) noexcept override;
  gxf::Expected<bool> isAcceptingRequest();

 private:
  gxf_result_t NvdiffrastRender(
    cudaStream_t cuda_stream, std::vector<Eigen::MatrixXf>& poses, Eigen::Matrix3f& K, Eigen::MatrixXf& bbox2d, int rgb_H,
    int rgb_W, int H, int W, nvcv::Tensor& flip_color_tensor, nvcv::Tensor& flip_xyz_map_tensor);

  gxf::Parameter<gxf::Handle<gxf::Receiver>> pose_receiver_;
  gxf::Parameter<gxf::Handle<gxf::Receiver>> rgb_receiver_;
  gxf::Parameter<gxf::Handle<gxf::Receiver>> camera_model_receiver_;
  gxf::Parameter<gxf::Handle<gxf::Receiver>> point_cloud_receiver_;
  gxf::Parameter<gxf::Handle<gxf::Receiver>> iterative_pose_receiver_;
  gxf::Parameter<gxf::Handle<gxf::Receiver>> iterative_rgb_receiver_;
  gxf::Parameter<gxf::Handle<gxf::Receiver>> iterative_camera_model_receiver_;
  gxf::Parameter<gxf::Handle<gxf::Receiver>> iterative_point_cloud_receiver_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> pose_array_transmitter_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> iterative_rgb_transmitter_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> iterative_camera_model_transmitter_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> iterative_point_cloud_transmitter_;

  gxf::Parameter<std::string> mesh_file_path_;
  gxf::Parameter<std::string> texture_path_;
  gxf::Parameter<std::string> mode_;
  gxf::Parameter<float> crop_ratio_;
  gxf::Parameter<float> min_depth_;
  gxf::Parameter<float> max_depth_;
  gxf::Parameter<uint32_t> resized_image_height_;
  gxf::Parameter<uint32_t> resized_image_width_;
  gxf::Parameter<int> refine_iterations_;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> allocator_;
  gxf::Parameter<gxf::Handle<gxf::CudaStreamPool>> cuda_stream_pool_;
  gxf::Handle<gxf::CudaStream> cuda_stream_handle_;

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mesh_vertices_;
  std::vector<int32_t> mesh_faces_;
  std::vector<float> texcoords_;
  std::vector<float> vertices_;

  int num_vertices_;
  int num_faces_;
  int num_texcoords_;
  int texture_map_height_;
  int texture_map_width_;
  float mesh_diameter_;

  // Input data (GPU) for rendering kernels
  float* pose_clip_device_;
  int32_t* mesh_faces_device_;
  float* texcoords_device_;
  float* pts_cam_device_;
  uint8_t* texture_map_device_;
  nvcv::Tensor float_texture_map_tensor_;

  // Output data (GPU) for rendering kernels
  float* rast_out_device_;
  float* texcoords_out_device_;
  float* color_device_;
  float* xyz_map_device_;
  float* transformed_rgb_device_;
  float* transformed_xyz_map_device_;
  nvcv::Tensor render_rgb_tensor_;
  nvcv::Tensor render_xyz_map_tensor_;
  std::unordered_map<int, float*> transformed_xyz_map_cache_;
  std::unordered_map<int, float*> transformed_rgb_cache_;

  float* score_rendered_output_device_;
  float* score_original_output_device_;
  bool render_data_cached_ = false;
  bool rgb_data_cached_ = false;
  CR::CudaRaster* cr_;

  int32_t score_recevied_batches_ = 0;
  int32_t refine_recevied_batches_ = 0;
  int32_t iteration_count_ = 0;

  cudaStream_t cuda_stream_ = 0;
};

}  // namespace isaac_ros
}  // namespace nvidia

#endif  // NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_RENDER_HPP_
