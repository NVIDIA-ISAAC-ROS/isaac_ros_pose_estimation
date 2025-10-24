// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <Eigen/Dense>

#include <string>
#include <vector>

#include "gxf/core/entity.hpp"
#include "gxf/core/gxf.h"
#include "gxf/core/parameter.hpp"
#include "gxf/core/parameter_parser_std.hpp"
#include "gxf/multimedia/camera.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/memory_buffer.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

#include "opencv2/core/mat.hpp"

#include "gxf/cuda/cuda_stream.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"

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
  gxf_result_t registerInterface(gxf::Registrar* registrar) noexcept override;

 private:
  gxf::Expected<void> updateCameraProperties(gxf::Handle<gxf::CameraModel> camera_model);
  gxf::Parameter<gxf::Handle<gxf::Receiver>> tensorlist_receiver_;
  gxf::Parameter<gxf::Handle<gxf::Receiver>> camera_model_input_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> detection3darray_transmitter_;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> allocator_;

  // Parameters
  gxf::Parameter<std::vector<double>> object_dimensions_param_;
  gxf::Parameter<std::vector<double>> camera_matrix_param_;
  gxf::Parameter<std::string> object_name_;
  gxf::Parameter<double> map_peak_threshold_;
  gxf::Parameter<double> affinity_map_angle_threshold_;
  // These params can be used to specify a rotation of the pose output by the network, it is useful
  // when one would like to align detections between Foundation pose and Dope. For example, for the
  // soup can asset, the rotation would need to done along the y axis by 90 degrees. ALl rotation
  // values here are in degrees. The rotation is performed in a ZYX sequence.
  gxf::Parameter<double> rotation_y_axis_;
  gxf::Parameter<double> rotation_x_axis_;
  gxf::Parameter<double> rotation_z_axis_;
  // CUDA stream pool
  gxf::Parameter<gxf::Handle<gxf::CudaStreamPool>> cuda_stream_pool_;

  // Parsed parameters
  Eigen::Matrix<double, 3, 9> cuboid_3d_points_;
  cv::Mat camera_matrix_;
  // CUDA stream variables
  gxf::Handle<gxf::CudaStream> cuda_stream_handle_;
  cudaStream_t cuda_stream_ = 0;
};

}  // namespace dope
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // NVIDIA_ISAAC_ROS_EXTENSIONS_DOPE_DECODER_HPP_
