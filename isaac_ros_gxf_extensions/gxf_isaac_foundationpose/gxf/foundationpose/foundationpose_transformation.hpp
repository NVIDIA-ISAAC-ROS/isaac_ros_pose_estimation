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

#ifndef NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_TRANSFORMATION_HPP_
#define NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_TRANSFORMATION_HPP_

#include "gxf/cuda/cuda_stream.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"
#include "gxf/core/entity.hpp"
#include "gxf/core/gxf.h"
#include "gxf/core/parameter.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/memory_buffer.hpp"
#include "gxf/core/parameter_parser_std.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

#include "mesh_storage.hpp"

namespace nvidia {
namespace isaac_ros {

// GXF codelet refine the pose estimations using the delta received from refine network
class FoundationposeTransformation : public gxf::Codelet {
 public:
  gxf_result_t start() noexcept override;
  gxf_result_t tick() noexcept override;
  gxf_result_t stop() noexcept override { return GXF_SUCCESS; }
  gxf_result_t registerInterface(gxf::Registrar* registrar) noexcept override;

 private:
  gxf::Parameter<gxf::Handle<gxf::Receiver>> poses_receiver_;
  gxf::Parameter<gxf::Handle<gxf::Receiver>> refined_poses_receiver_;
  gxf::Parameter<gxf::Handle<gxf::Receiver>> iterative_poses_receiver_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> iterative_sliced_pose_array_transmitter_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> sliced_pose_array_transmitter_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> batched_pose_array_transmitter_;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> allocator_;
  gxf::Parameter<gxf::Handle<gxf::CudaStreamPool>> cuda_stream_pool_;
  gxf::Parameter<gxf::Handle<MeshStorage>> mesh_storage_;
  gxf::Parameter<float> rot_normalizer_;
  gxf::Parameter<std::string> mode_;
  gxf::Parameter<int> refine_iterations_;
  gxf::Handle<gxf::CudaStream> cuda_stream_handle_;

  int32_t received_batches_ = 0;
  int32_t iteration_count_ = 0;
  std::vector<float> batched_refined_pose_;

  cudaStream_t cuda_stream_ = 0;
};

}  // namespace isaac_ros
}  // namespace nvidia

#endif  // NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_TRANSFORMATION_HPP_
