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

#define RAST_CUDA_FWD_SHADER_KERNEL_BLOCK_WIDTH  8
#define RAST_CUDA_FWD_SHADER_KERNEL_BLOCK_HEIGHT 8
#define RAST_GRAD_MAX_KERNEL_BLOCK_WIDTH  8
#define RAST_GRAD_MAX_KERNEL_BLOCK_HEIGHT 8

//------------------------------------------------------------------------
// CUDA forward rasterizer shader kernel params.

struct RasterizeCudaFwdShaderParams
{
    const float*    pos;            // Vertex positions.
    const int*      tri;            // Triangle indices.
    const int*      in_idx;         // Triangle idx buffer from rasterizer.
    float*          out;            // Main output buffer.
    int             numTriangles;   // Number of triangles.
    int             numVertices;    // Number of vertices.
    int             width;          // Image width.
    int             height;         // Image height.
    int             depth;          // Size of minibatch.
    int             instance_mode;  // 1 if in instance rendering mode.
    float           xs, xo, ys, yo; // Pixel position to clip-space x, y transform.
};

//------------------------------------------------------------------------
