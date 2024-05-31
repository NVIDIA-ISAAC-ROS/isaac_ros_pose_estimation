// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "extensions/centerpose/components/centerpose_detection_to_isaac.hpp"
#include "extensions/centerpose/components/centerpose_postprocessor.hpp"
#include "extensions/centerpose/components/centerpose_visualizer.hpp"
#include "detection3_d_array_message.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(
    0xc4091ff8477411ee, 0xa4e6036b2c3e1879, "CenterPose",
    "Extension containing CenterPose pose estimation related components", "Isaac SDK", "2.0.0",
    "LICENSE");

GXF_EXT_FACTORY_ADD(
    0xdaf98ba8477411ee, 0x93ba3f47723d071c, nvidia::isaac::centerpose::CenterPosePostProcessor,
    nvidia::gxf::Codelet,
    "Generates pose estimations given an output from the CenterPose neural network");

GXF_EXT_FACTORY_ADD(
    0x6c40d78e528c11ee, 0x990daf0e3d1195a0, nvidia::isaac::centerpose::CenterPoseDetectionToIsaac,
    nvidia::gxf::Codelet,
    "Converts pose estimations from the CenterPosePostProcessor into a Isaac-friendly format");

GXF_EXT_FACTORY_ADD(
    0x0e92a75e617111ee, 0x8c97b3583a2280c4, nvidia::isaac::centerpose::CenterPoseVisualizer,
    nvidia::gxf::Codelet,
    "Visualizes results of centerpose detections by projecting it onto images");

GXF_EXT_FACTORY_ADD_0(
    0x65d4476051a511ee, 0xb3f727d2ed955144, nvidia::isaac::ObjectHypothesis,
    "List of scores and class ids");

GXF_EXT_FACTORY_ADD_0(
    0x782823c851dc11ee, 0xa5dc87b46496e7b8, nvidia::isaac::Vector3f, "3 Dimensional Vector");

GXF_EXT_FACTORY_END()
