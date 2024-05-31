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
#include <cuda.h>
#include <stdint.h>

//------------------------------------------------------------------------
// C++ helper function prototypes.

dim3 getLaunchBlockSize(int maxWidth, int maxHeight, int width, int height);
dim3 getLaunchGridSize(dim3 blockSize, int width, int height, int depth);

//------------------------------------------------------------------------
// The rest is CUDA device code specific stuff.

#ifdef __CUDACC__

//------------------------------------------------------------------------
// Helpers for CUDA vector types.

static __device__ __forceinline__ float2&   operator*=  (float2& a, const float2& b)       { a.x *= b.x; a.y *= b.y; return a; }
static __device__ __forceinline__ float2&   operator+=  (float2& a, const float2& b)       { a.x += b.x; a.y += b.y; return a; }
static __device__ __forceinline__ float2&   operator-=  (float2& a, const float2& b)       { a.x -= b.x; a.y -= b.y; return a; }
static __device__ __forceinline__ float2&   operator*=  (float2& a, float b)               { a.x *= b; a.y *= b; return a; }
static __device__ __forceinline__ float2&   operator+=  (float2& a, float b)               { a.x += b; a.y += b; return a; }
static __device__ __forceinline__ float2&   operator-=  (float2& a, float b)               { a.x -= b; a.y -= b; return a; }
static __device__ __forceinline__ float2    operator*   (const float2& a, const float2& b) { return make_float2(a.x * b.x, a.y * b.y); }
static __device__ __forceinline__ float2    operator+   (const float2& a, const float2& b) { return make_float2(a.x + b.x, a.y + b.y); }
static __device__ __forceinline__ float2    operator-   (const float2& a, const float2& b) { return make_float2(a.x - b.x, a.y - b.y); }
static __device__ __forceinline__ float2    operator*   (const float2& a, float b)         { return make_float2(a.x * b, a.y * b); }
static __device__ __forceinline__ float2    operator+   (const float2& a, float b)         { return make_float2(a.x + b, a.y + b); }
static __device__ __forceinline__ float2    operator-   (const float2& a, float b)         { return make_float2(a.x - b, a.y - b); }
static __device__ __forceinline__ float2    operator*   (float a, const float2& b)         { return make_float2(a * b.x, a * b.y); }
static __device__ __forceinline__ float2    operator+   (float a, const float2& b)         { return make_float2(a + b.x, a + b.y); }
static __device__ __forceinline__ float2    operator-   (float a, const float2& b)         { return make_float2(a - b.x, a - b.y); }
static __device__ __forceinline__ float2    operator-   (const float2& a)                  { return make_float2(-a.x, -a.y); }
static __device__ __forceinline__ float3&   operator*=  (float3& a, const float3& b)       { a.x *= b.x; a.y *= b.y; a.z *= b.z; return a; }
static __device__ __forceinline__ float3&   operator+=  (float3& a, const float3& b)       { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
static __device__ __forceinline__ float3&   operator-=  (float3& a, const float3& b)       { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
static __device__ __forceinline__ float3&   operator*=  (float3& a, float b)               { a.x *= b; a.y *= b; a.z *= b; return a; }
static __device__ __forceinline__ float3&   operator+=  (float3& a, float b)               { a.x += b; a.y += b; a.z += b; return a; }
static __device__ __forceinline__ float3&   operator-=  (float3& a, float b)               { a.x -= b; a.y -= b; a.z -= b; return a; }
static __device__ __forceinline__ float3    operator*   (const float3& a, const float3& b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
static __device__ __forceinline__ float3    operator+   (const float3& a, const float3& b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
static __device__ __forceinline__ float3    operator-   (const float3& a, const float3& b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
static __device__ __forceinline__ float3    operator*   (const float3& a, float b)         { return make_float3(a.x * b, a.y * b, a.z * b); }
static __device__ __forceinline__ float3    operator+   (const float3& a, float b)         { return make_float3(a.x + b, a.y + b, a.z + b); }
static __device__ __forceinline__ float3    operator-   (const float3& a, float b)         { return make_float3(a.x - b, a.y - b, a.z - b); }
static __device__ __forceinline__ float3    operator*   (float a, const float3& b)         { return make_float3(a * b.x, a * b.y, a * b.z); }
static __device__ __forceinline__ float3    operator+   (float a, const float3& b)         { return make_float3(a + b.x, a + b.y, a + b.z); }
static __device__ __forceinline__ float3    operator-   (float a, const float3& b)         { return make_float3(a - b.x, a - b.y, a - b.z); }
static __device__ __forceinline__ float3    operator-   (const float3& a)                  { return make_float3(-a.x, -a.y, -a.z); }
static __device__ __forceinline__ float4&   operator*=  (float4& a, const float4& b)       { a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w; return a; }
static __device__ __forceinline__ float4&   operator+=  (float4& a, const float4& b)       { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; }
static __device__ __forceinline__ float4&   operator-=  (float4& a, const float4& b)       { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a; }
static __device__ __forceinline__ float4&   operator*=  (float4& a, float b)               { a.x *= b; a.y *= b; a.z *= b; a.w *= b; return a; }
static __device__ __forceinline__ float4&   operator+=  (float4& a, float b)               { a.x += b; a.y += b; a.z += b; a.w += b; return a; }
static __device__ __forceinline__ float4&   operator-=  (float4& a, float b)               { a.x -= b; a.y -= b; a.z -= b; a.w -= b; return a; }
static __device__ __forceinline__ float4    operator*   (const float4& a, const float4& b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
static __device__ __forceinline__ float4    operator+   (const float4& a, const float4& b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
static __device__ __forceinline__ float4    operator-   (const float4& a, const float4& b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
static __device__ __forceinline__ float4    operator*   (const float4& a, float b)         { return make_float4(a.x * b, a.y * b, a.z * b, a.w * b); }
static __device__ __forceinline__ float4    operator+   (const float4& a, float b)         { return make_float4(a.x + b, a.y + b, a.z + b, a.w + b); }
static __device__ __forceinline__ float4    operator-   (const float4& a, float b)         { return make_float4(a.x - b, a.y - b, a.z - b, a.w - b); }
static __device__ __forceinline__ float4    operator*   (float a, const float4& b)         { return make_float4(a * b.x, a * b.y, a * b.z, a * b.w); }
static __device__ __forceinline__ float4    operator+   (float a, const float4& b)         { return make_float4(a + b.x, a + b.y, a + b.z, a + b.w); }
static __device__ __forceinline__ float4    operator-   (float a, const float4& b)         { return make_float4(a - b.x, a - b.y, a - b.z, a - b.w); }
static __device__ __forceinline__ float4    operator-   (const float4& a)                  { return make_float4(-a.x, -a.y, -a.z, -a.w); }
static __device__ __forceinline__ int2&     operator*=  (int2& a, const int2& b)           { a.x *= b.x; a.y *= b.y; return a; }
static __device__ __forceinline__ int2&     operator+=  (int2& a, const int2& b)           { a.x += b.x; a.y += b.y; return a; }
static __device__ __forceinline__ int2&     operator-=  (int2& a, const int2& b)           { a.x -= b.x; a.y -= b.y; return a; }
static __device__ __forceinline__ int2&     operator*=  (int2& a, int b)                   { a.x *= b; a.y *= b; return a; }
static __device__ __forceinline__ int2&     operator+=  (int2& a, int b)                   { a.x += b; a.y += b; return a; }
static __device__ __forceinline__ int2&     operator-=  (int2& a, int b)                   { a.x -= b; a.y -= b; return a; }
static __device__ __forceinline__ int2      operator*   (const int2& a, const int2& b)     { return make_int2(a.x * b.x, a.y * b.y); }
static __device__ __forceinline__ int2      operator+   (const int2& a, const int2& b)     { return make_int2(a.x + b.x, a.y + b.y); }
static __device__ __forceinline__ int2      operator-   (const int2& a, const int2& b)     { return make_int2(a.x - b.x, a.y - b.y); }
static __device__ __forceinline__ int2      operator*   (const int2& a, int b)             { return make_int2(a.x * b, a.y * b); }
static __device__ __forceinline__ int2      operator+   (const int2& a, int b)             { return make_int2(a.x + b, a.y + b); }
static __device__ __forceinline__ int2      operator-   (const int2& a, int b)             { return make_int2(a.x - b, a.y - b); }
static __device__ __forceinline__ int2      operator*   (int a, const int2& b)             { return make_int2(a * b.x, a * b.y); }
static __device__ __forceinline__ int2      operator+   (int a, const int2& b)             { return make_int2(a + b.x, a + b.y); }
static __device__ __forceinline__ int2      operator-   (int a, const int2& b)             { return make_int2(a - b.x, a - b.y); }
static __device__ __forceinline__ int2      operator-   (const int2& a)                    { return make_int2(-a.x, -a.y); }
static __device__ __forceinline__ int3&     operator*=  (int3& a, const int3& b)           { a.x *= b.x; a.y *= b.y; a.z *= b.z; return a; }
static __device__ __forceinline__ int3&     operator+=  (int3& a, const int3& b)           { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
static __device__ __forceinline__ int3&     operator-=  (int3& a, const int3& b)           { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
static __device__ __forceinline__ int3&     operator*=  (int3& a, int b)                   { a.x *= b; a.y *= b; a.z *= b; return a; }
static __device__ __forceinline__ int3&     operator+=  (int3& a, int b)                   { a.x += b; a.y += b; a.z += b; return a; }
static __device__ __forceinline__ int3&     operator-=  (int3& a, int b)                   { a.x -= b; a.y -= b; a.z -= b; return a; }
static __device__ __forceinline__ int3      operator*   (const int3& a, const int3& b)     { return make_int3(a.x * b.x, a.y * b.y, a.z * b.z); }
static __device__ __forceinline__ int3      operator+   (const int3& a, const int3& b)     { return make_int3(a.x + b.x, a.y + b.y, a.z + b.z); }
static __device__ __forceinline__ int3      operator-   (const int3& a, const int3& b)     { return make_int3(a.x - b.x, a.y - b.y, a.z - b.z); }
static __device__ __forceinline__ int3      operator*   (const int3& a, int b)             { return make_int3(a.x * b, a.y * b, a.z * b); }
static __device__ __forceinline__ int3      operator+   (const int3& a, int b)             { return make_int3(a.x + b, a.y + b, a.z + b); }
static __device__ __forceinline__ int3      operator-   (const int3& a, int b)             { return make_int3(a.x - b, a.y - b, a.z - b); }
static __device__ __forceinline__ int3      operator*   (int a, const int3& b)             { return make_int3(a * b.x, a * b.y, a * b.z); }
static __device__ __forceinline__ int3      operator+   (int a, const int3& b)             { return make_int3(a + b.x, a + b.y, a + b.z); }
static __device__ __forceinline__ int3      operator-   (int a, const int3& b)             { return make_int3(a - b.x, a - b.y, a - b.z); }
static __device__ __forceinline__ int3      operator-   (const int3& a)                    { return make_int3(-a.x, -a.y, -a.z); }
static __device__ __forceinline__ int4&     operator*=  (int4& a, const int4& b)           { a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w; return a; }
static __device__ __forceinline__ int4&     operator+=  (int4& a, const int4& b)           { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; }
static __device__ __forceinline__ int4&     operator-=  (int4& a, const int4& b)           { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a; }
static __device__ __forceinline__ int4&     operator*=  (int4& a, int b)                   { a.x *= b; a.y *= b; a.z *= b; a.w *= b; return a; }
static __device__ __forceinline__ int4&     operator+=  (int4& a, int b)                   { a.x += b; a.y += b; a.z += b; a.w += b; return a; }
static __device__ __forceinline__ int4&     operator-=  (int4& a, int b)                   { a.x -= b; a.y -= b; a.z -= b; a.w -= b; return a; }
static __device__ __forceinline__ int4      operator*   (const int4& a, const int4& b)     { return make_int4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
static __device__ __forceinline__ int4      operator+   (const int4& a, const int4& b)     { return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
static __device__ __forceinline__ int4      operator-   (const int4& a, const int4& b)     { return make_int4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
static __device__ __forceinline__ int4      operator*   (const int4& a, int b)             { return make_int4(a.x * b, a.y * b, a.z * b, a.w * b); }
static __device__ __forceinline__ int4      operator+   (const int4& a, int b)             { return make_int4(a.x + b, a.y + b, a.z + b, a.w + b); }
static __device__ __forceinline__ int4      operator-   (const int4& a, int b)             { return make_int4(a.x - b, a.y - b, a.z - b, a.w - b); }
static __device__ __forceinline__ int4      operator*   (int a, const int4& b)             { return make_int4(a * b.x, a * b.y, a * b.z, a * b.w); }
static __device__ __forceinline__ int4      operator+   (int a, const int4& b)             { return make_int4(a + b.x, a + b.y, a + b.z, a + b.w); }
static __device__ __forceinline__ int4      operator-   (int a, const int4& b)             { return make_int4(a - b.x, a - b.y, a - b.z, a - b.w); }
static __device__ __forceinline__ int4      operator-   (const int4& a)                    { return make_int4(-a.x, -a.y, -a.z, -a.w); }
static __device__ __forceinline__ uint2&    operator*=  (uint2& a, const uint2& b)         { a.x *= b.x; a.y *= b.y; return a; }
static __device__ __forceinline__ uint2&    operator+=  (uint2& a, const uint2& b)         { a.x += b.x; a.y += b.y; return a; }
static __device__ __forceinline__ uint2&    operator-=  (uint2& a, const uint2& b)         { a.x -= b.x; a.y -= b.y; return a; }
static __device__ __forceinline__ uint2&    operator*=  (uint2& a, unsigned int b)         { a.x *= b; a.y *= b; return a; }
static __device__ __forceinline__ uint2&    operator+=  (uint2& a, unsigned int b)         { a.x += b; a.y += b; return a; }
static __device__ __forceinline__ uint2&    operator-=  (uint2& a, unsigned int b)         { a.x -= b; a.y -= b; return a; }
static __device__ __forceinline__ uint2     operator*   (const uint2& a, const uint2& b)   { return make_uint2(a.x * b.x, a.y * b.y); }
static __device__ __forceinline__ uint2     operator+   (const uint2& a, const uint2& b)   { return make_uint2(a.x + b.x, a.y + b.y); }
static __device__ __forceinline__ uint2     operator-   (const uint2& a, const uint2& b)   { return make_uint2(a.x - b.x, a.y - b.y); }
static __device__ __forceinline__ uint2     operator*   (const uint2& a, unsigned int b)   { return make_uint2(a.x * b, a.y * b); }
static __device__ __forceinline__ uint2     operator+   (const uint2& a, unsigned int b)   { return make_uint2(a.x + b, a.y + b); }
static __device__ __forceinline__ uint2     operator-   (const uint2& a, unsigned int b)   { return make_uint2(a.x - b, a.y - b); }
static __device__ __forceinline__ uint2     operator*   (unsigned int a, const uint2& b)   { return make_uint2(a * b.x, a * b.y); }
static __device__ __forceinline__ uint2     operator+   (unsigned int a, const uint2& b)   { return make_uint2(a + b.x, a + b.y); }
static __device__ __forceinline__ uint2     operator-   (unsigned int a, const uint2& b)   { return make_uint2(a - b.x, a - b.y); }
static __device__ __forceinline__ uint3&    operator*=  (uint3& a, const uint3& b)         { a.x *= b.x; a.y *= b.y; a.z *= b.z; return a; }
static __device__ __forceinline__ uint3&    operator+=  (uint3& a, const uint3& b)         { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
static __device__ __forceinline__ uint3&    operator-=  (uint3& a, const uint3& b)         { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
static __device__ __forceinline__ uint3&    operator*=  (uint3& a, unsigned int b)         { a.x *= b; a.y *= b; a.z *= b; return a; }
static __device__ __forceinline__ uint3&    operator+=  (uint3& a, unsigned int b)         { a.x += b; a.y += b; a.z += b; return a; }
static __device__ __forceinline__ uint3&    operator-=  (uint3& a, unsigned int b)         { a.x -= b; a.y -= b; a.z -= b; return a; }
static __device__ __forceinline__ uint3     operator*   (const uint3& a, const uint3& b)   { return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z); }
static __device__ __forceinline__ uint3     operator+   (const uint3& a, const uint3& b)   { return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z); }
static __device__ __forceinline__ uint3     operator-   (const uint3& a, const uint3& b)   { return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z); }
static __device__ __forceinline__ uint3     operator*   (const uint3& a, unsigned int b)   { return make_uint3(a.x * b, a.y * b, a.z * b); }
static __device__ __forceinline__ uint3     operator+   (const uint3& a, unsigned int b)   { return make_uint3(a.x + b, a.y + b, a.z + b); }
static __device__ __forceinline__ uint3     operator-   (const uint3& a, unsigned int b)   { return make_uint3(a.x - b, a.y - b, a.z - b); }
static __device__ __forceinline__ uint3     operator*   (unsigned int a, const uint3& b)   { return make_uint3(a * b.x, a * b.y, a * b.z); }
static __device__ __forceinline__ uint3     operator+   (unsigned int a, const uint3& b)   { return make_uint3(a + b.x, a + b.y, a + b.z); }
static __device__ __forceinline__ uint3     operator-   (unsigned int a, const uint3& b)   { return make_uint3(a - b.x, a - b.y, a - b.z); }
static __device__ __forceinline__ uint4&    operator*=  (uint4& a, const uint4& b)         { a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w; return a; }
static __device__ __forceinline__ uint4&    operator+=  (uint4& a, const uint4& b)         { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; }
static __device__ __forceinline__ uint4&    operator-=  (uint4& a, const uint4& b)         { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a; }
static __device__ __forceinline__ uint4&    operator*=  (uint4& a, unsigned int b)         { a.x *= b; a.y *= b; a.z *= b; a.w *= b; return a; }
static __device__ __forceinline__ uint4&    operator+=  (uint4& a, unsigned int b)         { a.x += b; a.y += b; a.z += b; a.w += b; return a; }
static __device__ __forceinline__ uint4&    operator-=  (uint4& a, unsigned int b)         { a.x -= b; a.y -= b; a.z -= b; a.w -= b; return a; }
static __device__ __forceinline__ uint4     operator*   (const uint4& a, const uint4& b)   { return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
static __device__ __forceinline__ uint4     operator+   (const uint4& a, const uint4& b)   { return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
static __device__ __forceinline__ uint4     operator-   (const uint4& a, const uint4& b)   { return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
static __device__ __forceinline__ uint4     operator*   (const uint4& a, unsigned int b)   { return make_uint4(a.x * b, a.y * b, a.z * b, a.w * b); }
static __device__ __forceinline__ uint4     operator+   (const uint4& a, unsigned int b)   { return make_uint4(a.x + b, a.y + b, a.z + b, a.w + b); }
static __device__ __forceinline__ uint4     operator-   (const uint4& a, unsigned int b)   { return make_uint4(a.x - b, a.y - b, a.z - b, a.w - b); }
static __device__ __forceinline__ uint4     operator*   (unsigned int a, const uint4& b)   { return make_uint4(a * b.x, a * b.y, a * b.z, a * b.w); }
static __device__ __forceinline__ uint4     operator+   (unsigned int a, const uint4& b)   { return make_uint4(a + b.x, a + b.y, a + b.z, a + b.w); }
static __device__ __forceinline__ uint4     operator-   (unsigned int a, const uint4& b)   { return make_uint4(a - b.x, a - b.y, a - b.z, a - b.w); }

template<class T> static __device__ __forceinline__ T zero_value(void);
template<> __device__ __forceinline__ float  zero_value<float> (void)                      { return 0.f; }
template<> __device__ __forceinline__ float2 zero_value<float2>(void)                      { return make_float2(0.f, 0.f); }
template<> __device__ __forceinline__ float4 zero_value<float4>(void)                      { return make_float4(0.f, 0.f, 0.f, 0.f); }
static __device__ __forceinline__ float3 make_float3(const float2& a, float b)             { return make_float3(a.x, a.y, b); }
static __device__ __forceinline__ float4 make_float4(const float3& a, float b)             { return make_float4(a.x, a.y, a.z, b); }
static __device__ __forceinline__ float4 make_float4(const float2& a, const float2& b)     { return make_float4(a.x, a.y, b.x, b.y); }
static __device__ __forceinline__ int3 make_int3(const int2& a, int b)                     { return make_int3(a.x, a.y, b); }
static __device__ __forceinline__ int4 make_int4(const int3& a, int b)                     { return make_int4(a.x, a.y, a.z, b); }
static __device__ __forceinline__ int4 make_int4(const int2& a, const int2& b)             { return make_int4(a.x, a.y, b.x, b.y); }
static __device__ __forceinline__ uint3 make_uint3(const uint2& a, unsigned int b)         { return make_uint3(a.x, a.y, b); }
static __device__ __forceinline__ uint4 make_uint4(const uint3& a, unsigned int b)         { return make_uint4(a.x, a.y, a.z, b); }
static __device__ __forceinline__ uint4 make_uint4(const uint2& a, const uint2& b)         { return make_uint4(a.x, a.y, b.x, b.y); }

template<class T> static __device__ __forceinline__ void swap(T& a, T& b)                  { T temp = a; a = b; b = temp; }

//------------------------------------------------------------------------
#endif // __CUDACC__
