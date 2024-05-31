// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

//------------------------------------------------------------------------
// Constants and helpers.

#define IP_FWD_MAX_KERNEL_BLOCK_WIDTH   8
#define IP_FWD_MAX_KERNEL_BLOCK_HEIGHT  8

//------------------------------------------------------------------------
// CUDA kernel params.

struct InterpolateKernelParams
{
    const int*      tri;                            // Incoming triangle buffer.
    const float*    attr;                           // Incoming attribute buffer.
    const float*    rast;                           // Incoming rasterizer output buffer.
    float*          out;                            // Outgoing interpolated attributes.
    int             numTriangles;                   // Number of triangles.
    int             numVertices;                    // Number of vertices.
    int             numAttr;                        // Number of total vertex attributes.
    int             width;                          // Image width.
    int             height;                         // Image height.
    int             depth;                          // Minibatch size.
    int             attrBC;                         // 0=normal, 1=attr is broadcast.
    int             instance_mode;                  // 0=normal, 1=instance mode.
};

//------------------------------------------------------------------------
