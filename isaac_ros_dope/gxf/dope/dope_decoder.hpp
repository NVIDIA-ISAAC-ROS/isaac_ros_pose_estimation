// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVIDIA_ISAAC_ROS_EXTENSIONS_DOPE_DECODER_HPP_
#define NVIDIA_ISAAC_ROS_EXTENSIONS_DOPE_DECODER_HPP_

#include "opencv2/core/mat.hpp"
#include <Eigen/Dense>

#include "gxf/core/entity.hpp"
#include "gxf/core/gxf.h"
#include "gxf/core/parameter.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/memory_buffer.hpp"
#include "gxf/std/parameter_parser_std.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace isaac_ros {
namespace dope {

// GXF codelet that decodes object poses from an input tensor and produces an
// output tensor representing a pose array
class DopeDecoder : public gxf::Codelet {
public:
  gxf_result_t start() noexcept override;
  gxf_result_t tick() noexcept override;
  gxf_result_t stop() noexcept override { return GXF_SUCCESS; }
  gxf_result_t registerInterface(gxf::Registrar *registrar) noexcept override;

private:
  gxf::Parameter<gxf::Handle<gxf::Receiver>> tensorlist_receiver_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> posearray_transmitter_;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> allocator_;

  // Parameters
  gxf::Parameter<std::vector<double>> object_dimensions_param_;
  gxf::Parameter<std::vector<double>> camera_matrix_param_;
  gxf::Parameter<std::string> object_name_;

  // Parsed parameters
  Eigen::Matrix<double, 3, 9> cuboid_3d_points_;
  cv::Mat camera_matrix_;
};

} // namespace dope
} // namespace isaac_ros
} // namespace nvidia

#endif // NVIDIA_ISAAC_ROS_EXTENSIONS_DOPE_DECODER_HPP_
