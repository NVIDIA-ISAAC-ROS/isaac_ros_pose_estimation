// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dope_decoder.hpp"

extern "C" {

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(0x331242ce182a11ed, 0x861d0242ac120002,
                         "DopeExtension", "DOPE GXF extension", "NVIDIA",
                         "1.0.0", "LICENSE");

GXF_EXT_FACTORY_ADD(0x9bd4f8c4182a11ed, 0x861d0242ac120002,
                    nvidia::isaac_ros::dope::DopeDecoder, nvidia::gxf::Codelet,
                    "Codelet to decode DOPE output.");

GXF_EXT_FACTORY_END()

} // extern "C"
