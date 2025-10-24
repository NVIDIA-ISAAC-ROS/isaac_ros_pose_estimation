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
#pragma once

#include <vector>

#include "Eigen/Dense"
#include "extensions/centerpose/components/centerpose_detection.hpp"
#include "extensions/centerpose/components/centerpose_types.hpp"
#include "extensions/centerpose/components/cuboid3d.hpp"
#include "extensions/centerpose/components/cuboid_pnp_solver.hpp"
#include "gxf/core/parameter_parser_std.hpp"
#include "gxf/multimedia/camera.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/timestamp.hpp"
#include "gxf/std/transmitter.hpp"

#include "gxf/cuda/cuda_stream.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"

namespace nvidia {
namespace isaac {
namespace centerpose {

class CenterPosePostProcessor : public gxf::Codelet {
 public:
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;

 private:
  CenterPoseDetectionList processTensor(const std::vector<Eigen::MatrixXfRM>& tensors);

  gxf::Expected<void> publish(
      const CenterPoseDetectionList& detections, gxf::Handle<gxf::Timestamp> input_timestamp);

  gxf::Expected<void> updateCameraProperties(gxf::Handle<gxf::CameraModel> camera_model);

 private:
  gxf::Parameter<gxf::Handle<gxf::Receiver>> input_;
  gxf::Parameter<gxf::Handle<gxf::Receiver>> camera_model_input_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> output_;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> allocator_;
  gxf::Parameter<std::vector<int>> output_field_size_param_;
  gxf::Parameter<float> cuboid_scaling_factor_;
  gxf::Parameter<int32_t> storage_type_;
  gxf::Parameter<float> score_threshold_;
  gxf::Parameter<gxf::Handle<gxf::CudaStreamPool>> cuda_stream_pool_;
  // CUDA stream variables
  gxf::Handle<gxf::CudaStream> cuda_stream_handle_;
  cudaStream_t cuda_stream_ = 0;

  Eigen::Matrix3f camera_matrix_;
  Eigen::Vector2i original_image_size_;
  Eigen::Vector2i output_field_size_;
  Eigen::Matrix3fRM affine_transform_;
};

}  // namespace centerpose
}  // namespace isaac
}  // namespace nvidia
