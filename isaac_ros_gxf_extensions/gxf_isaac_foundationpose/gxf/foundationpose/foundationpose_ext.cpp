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
#include <string>

#include "gxf/core/gxf.h"
#include "gxf/std/extension_factory_helper.hpp"

#include "detection3_d_array_message/detection3_d_array_message.hpp"
#include "foundationpose_decoder.hpp"
#include "foundationpose_render.hpp"
#include "foundationpose_sampling.hpp"
#include "foundationpose_sync.hpp"
#include "foundationpose_transformation.hpp"
#include "mesh_storage.hpp"

extern "C" {

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(
    0x5b942ff659bc4502, 0xa0b000b36b53f74f, "FoundationPoseExtension",
    "FoundationPose GXF extension", "NVIDIA", "1.0.0", "LICENSE");

GXF_EXT_FACTORY_ADD(
    0xf6bbcfd111984aeb, 0xb6dbf26d3125540f, nvidia::isaac_ros::MeshStorage,
    nvidia::gxf::Component, "Stores and manages mesh data");

GXF_EXT_FACTORY_ADD(
    0x52138759e5824999, 0x8d26bd90e602d335, nvidia::isaac_ros::FoundationposeSampling,
    nvidia::gxf::Codelet, "Codelet to generate pose hypothesis.");

GXF_EXT_FACTORY_ADD(
    0xb2d0e3737b4f583f, 0xa13437c086f114e3, nvidia::isaac_ros::FoundationposeDecoder,
    nvidia::gxf::Codelet, "Codelet to decode, sort and select the final pose.");

GXF_EXT_FACTORY_ADD(
    0x52138759e5824998, 0x8d26bd90e603d335, nvidia::isaac_ros::FoundationPoseSynchronization,
    nvidia::gxf::Codelet, "Codelet to generate sync pairs for Foundation pose loads");

GXF_EXT_FACTORY_ADD(
    0x97104bb7e89f2b2a, 0x9de6f7fd399a201d, nvidia::isaac_ros::FoundationposeTransformation,
    nvidia::gxf::Codelet, "Codelet to trasnform poses for foundation pose.");

GXF_EXT_FACTORY_ADD(
    0x63138759e5624879, 0x4a95bd90e602d324, nvidia::isaac_ros::FoundationposeRender,
    nvidia::gxf::Codelet, "Codelet to generate pose hypothesis.");

GXF_EXT_FACTORY_ADD_0(
    0x65d4476051a512ee, 0xb3f727d2ed955145, nvidia::isaac::ObjectHypothesis,
    "List of scores and class ids");

GXF_EXT_FACTORY_ADD_0(
    0x782823c851dc11ea, 0xa5dc87b46496e7bd, nvidia::isaac::Vector3f, "3 Dimensional Vector");

GXF_EXT_FACTORY_END()

}  // extern "C"
