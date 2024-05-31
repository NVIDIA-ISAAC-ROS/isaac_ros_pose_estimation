// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2009-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cuda_runtime.h>
#include <cstdint>

namespace CR
{
//------------------------------------------------------------------------

#ifndef NULL
#   define NULL 0
#endif

#ifdef __CUDACC__
#   define CR_CUDA 1
#else
#   define CR_CUDA 0
#endif

#if CR_CUDA
#   define CR_CUDA_FUNC     __device__ __inline__
#   define CR_CUDA_CONST    __constant__
#else
#   define CR_CUDA_FUNC     inline
#   define CR_CUDA_CONST    static const
#endif

#define CR_UNREF(X)         ((void)(X))
#define CR_ARRAY_SIZE(X)    ((int)(sizeof(X) / sizeof((X)[0])))

//------------------------------------------------------------------------

typedef uint8_t             U8;
typedef uint16_t            U16;
typedef uint32_t            U32;
typedef uint64_t            U64;
typedef int8_t              S8;
typedef int16_t             S16;
typedef int32_t             S32;
typedef int64_t             S64;
typedef float               F32;
typedef double              F64;
typedef void                (*FuncPtr)(void);

//------------------------------------------------------------------------

#define CR_U32_MAX          (0xFFFFFFFFu)
#define CR_S32_MIN          (~0x7FFFFFFF)
#define CR_S32_MAX          (0x7FFFFFFF)
#define CR_U64_MAX          ((U64)(S64)-1)
#define CR_S64_MIN          ((S64)-1 << 63)
#define CR_S64_MAX          (~((S64)-1 << 63))
#define CR_F32_MIN          (1.175494351e-38f)
#define CR_F32_MAX          (3.402823466e+38f)
#define CR_F64_MIN          (2.2250738585072014e-308)
#define CR_F64_MAX          (1.7976931348623158e+308)

//------------------------------------------------------------------------
// Misc types.

class Vec2i
{
public:
    Vec2i(int x_, int y_) : x(x_), y(y_) {}
    int x, y;
};

class Vec3i
{
public:
    Vec3i(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
    int x, y, z;
};

//------------------------------------------------------------------------
// CUDA utilities.

#if CR_CUDA
#   define globalThreadIdx (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y)))
#endif

//------------------------------------------------------------------------
} // namespace CR
