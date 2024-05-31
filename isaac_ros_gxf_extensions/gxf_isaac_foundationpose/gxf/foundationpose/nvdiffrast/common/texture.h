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
#include "framework.h"

//------------------------------------------------------------------------
// Constants.

#define TEX_FWD_MAX_KERNEL_BLOCK_WIDTH          8
#define TEX_FWD_MAX_KERNEL_BLOCK_HEIGHT         8
#define TEX_FWD_MAX_MIP_KERNEL_BLOCK_WIDTH      8
#define TEX_FWD_MAX_MIP_KERNEL_BLOCK_HEIGHT     8
#define TEX_GRAD_MAX_KERNEL_BLOCK_WIDTH         8
#define TEX_GRAD_MAX_KERNEL_BLOCK_HEIGHT        8
#define TEX_GRAD_MAX_MIP_KERNEL_BLOCK_WIDTH     8
#define TEX_GRAD_MAX_MIP_KERNEL_BLOCK_HEIGHT    8
#define TEX_MAX_MIP_LEVEL                       16  // Currently a texture cannot be larger than 2 GB because we use 32-bit indices everywhere.
#define TEX_MODE_NEAREST                        0   // Nearest on base level.
#define TEX_MODE_LINEAR                         1   // Bilinear on base level.
#define TEX_MODE_LINEAR_MIPMAP_NEAREST          2   // Bilinear on nearest mip level.
#define TEX_MODE_LINEAR_MIPMAP_LINEAR           3   // Trilinear.
#define TEX_MODE_COUNT                          4
#define TEX_BOUNDARY_MODE_CUBE                  0   // Cube map mode.
#define TEX_BOUNDARY_MODE_WRAP                  1   // Wrap (u, v).
#define TEX_BOUNDARY_MODE_CLAMP                 2   // Clamp (u, v).
#define TEX_BOUNDARY_MODE_ZERO                  3   // Pad with zeros.
#define TEX_BOUNDARY_MODE_COUNT                 4

//------------------------------------------------------------------------
// CUDA kernel params.

struct TextureKernelParams
{
    const float*    tex[TEX_MAX_MIP_LEVEL];         // Incoming texture buffer with mip levels.
    const float*    uv;                             // Incoming texcoord buffer.
    const float*    mipLevelBias;                   // Incoming mip level bias or NULL.
    float*          out;                            // Outgoing texture data.
    float*          gradTex[TEX_MAX_MIP_LEVEL];     // Outgoing texture gradients with mip levels.
    float*          gradUV;                         // Outgoing texcoord gradient.
    float*          gradUVDA;                       // Outgoing texcoord pixel differential gradient.
    float*          gradMipLevelBias;               // Outgoing mip level bias gradient.
    int             enableMip;                      // If true, we have uv_da and/or mip_level_bias input(s), and a mip tensor.
    int             filterMode;                     // One of the TEX_MODE_ constants.
    int             boundaryMode;                   // One of the TEX_BOUNDARY_MODE_ contants.
    int             texConst;                       // If true, texture is known to be constant.
    int             mipLevelLimit;                  // Mip level limit coming from the op.
    int             channels;                       // Number of texture channels.
    int             imgWidth;                       // Image width.
    int             imgHeight;                      // Image height.
    int             texWidth;                       // Texture width.
    int             texHeight;                      // Texture height.
    int             texDepth;                       // Texture depth.
    int             n;                              // Minibatch size.
    int             mipLevelMax;                    // Maximum mip level index. Zero if mips disabled.
    int             mipLevelOut;                    // Mip level being calculated in builder kernel.
};

//------------------------------------------------------------------------
// C++ helper function prototypes.

// ISAAC ROS patch
// void raiseMipSizeError(NVDR_CTX_ARGS, const TextureKernelParams& p);
// int calculateMipInfo(NVDR_CTX_ARGS, TextureKernelParams& p, int* mipOffsets);

//------------------------------------------------------------------------
// Macros.

#define mipLevelSize(p, i) make_int2(((p).texWidth >> (i)) > 1 ? ((p).texWidth >> (i)) : 1, ((p).texHeight >> (i)) > 1 ? ((p).texHeight >> (i)) : 1)

//------------------------------------------------------------------------
