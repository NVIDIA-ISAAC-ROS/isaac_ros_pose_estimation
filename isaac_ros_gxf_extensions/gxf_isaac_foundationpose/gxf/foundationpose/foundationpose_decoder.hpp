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

#ifndef NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_DECODER_HPP_
#define NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_DECODER_HPP_

#include <Eigen/Dense>
#include <string>

#include "gxf/cuda/cuda_stream.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"
#include "gxf/core/entity.hpp"
#include "gxf/core/gxf.h"
#include "gxf/core/parameter.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

#include "mesh_storage.hpp"

namespace nvidia {
namespace isaac_ros {

// GXF codelet that selects the pose with the highest score
// and outputs its as a Detection3DListMessage
class FoundationposeDecoder : public gxf::Codelet {
 public:
  gxf_result_t start() noexcept override;
  gxf_result_t tick() noexcept override;
  gxf_result_t stop() noexcept override { return GXF_SUCCESS; }
  gxf_result_t registerInterface(gxf::Registrar* registrar) noexcept override;

 private:
  gxf::Parameter<gxf::Handle<gxf::Receiver>> pose_array_receiver_;
  gxf::Parameter<gxf::Handle<gxf::Receiver>> pose_scores_receiver_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> detection3_d_list_transmitter_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> pose_matrix_transmitter_;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> allocator_;
  gxf::Parameter<gxf::Handle<gxf::CudaStreamPool>> cuda_stream_pool_;
  gxf::Parameter<std::string> mode_;
  gxf::Handle<gxf::CudaStream> cuda_stream_handle_;
  gxf::Parameter<gxf::Handle<MeshStorage>> mesh_storage_;

  Eigen::Vector3f mesh_model_center_;
  cudaStream_t cuda_stream_ = 0;

  gxf_result_t ExtractBoundingBoxDimensions();
};

}  // namespace isaac_ros
}  // namespace nvidia

#endif  // NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_DECODER_HPP_
