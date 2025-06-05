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

#ifndef NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_SAMPLING_HPP_
#define NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_SAMPLING_HPP_

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "gxf/cuda/cuda_stream.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"
#include "gxf/core/entity.hpp"
#include "gxf/core/gxf.h"
#include "gxf/core/parameter.hpp"
#include "gxf/core/parameter_parser_std.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/memory_buffer.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

#include "foundationpose_sampling.cu.hpp"
#include "foundationpose_utils.hpp"
#include "mesh_storage.hpp"

namespace nvidia {
namespace isaac_ros {

// GXF codelet generate initial pose estimations on an icosphere.
class FoundationposeSampling : public gxf::Codelet {
 public:
  gxf_result_t start() noexcept override;
  gxf_result_t tick() noexcept override;
  gxf_result_t stop() noexcept override;
  gxf_result_t registerInterface(gxf::Registrar* registrar) noexcept override;

 private:
  gxf::Parameter<gxf::Handle<gxf::Receiver>> point_cloud_receiver_;
  gxf::Parameter<gxf::Handle<gxf::Receiver>> depth_receiver_;
  gxf::Parameter<gxf::Handle<gxf::Receiver>> rgb_receiver_;
  gxf::Parameter<gxf::Handle<gxf::Receiver>> segmentation_receiver_;

  gxf::Parameter<gxf::Handle<gxf::Transmitter>> posearray_transmitter_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> point_cloud_transmitter_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> rgb_transmitter_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> camera_model_transmitter_;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> allocator_;
  gxf::Parameter<gxf::Handle<gxf::CudaStreamPool>> cuda_stream_pool_;
  gxf::Parameter<uint32_t> max_hypothesis_;
  gxf::Parameter<float> min_depth_;
  gxf::Parameter<std::vector<std::string>> symmetry_axes_;
  gxf::Parameter<std::vector<std::string>> symmetry_planes_;
  gxf::Parameter<std::vector<std::string>> fixed_axis_angles_;
  gxf::Parameter<std::vector<std::string>> fixed_translations_;
  gxf::Handle<gxf::CudaStream> cuda_stream_handle_;
  gxf::Parameter<gxf::Handle<MeshStorage>> mesh_storage_;

  float* erode_depth_device_;
  float* bilateral_filter_depth_device_;
  bool cached_ = false;

  cudaStream_t cuda_stream_ = 0;
};

}  // namespace isaac_ros
}  // namespace nvidia

#endif  // NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_SAMPLING_HPP_
