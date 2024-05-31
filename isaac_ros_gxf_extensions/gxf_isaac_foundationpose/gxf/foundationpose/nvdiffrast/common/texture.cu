// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "common.h"
#include "texture.h"

//------------------------------------------------------------------------
// Memory access and math helpers.

template<class T> static __device__ __forceinline__ T lerp  (const T& a, const T& b, float c) { return a + c * (b - a); }
template<class T> static __device__ __forceinline__ T bilerp(const T& a, const T& b, const T& c, const T& d, const float2& e) { return lerp(lerp(a, b, e.x), lerp(c, d, e.x), e.y); }


template <bool CUBE_MODE>
static __device__ __forceinline__ float2 indexTextureLinear(const TextureKernelParams& p, float3 uv, int tz, int4& tcOut, int level)
{
    // Mip level size.
    int2 sz = mipLevelSize(p, level);
    int w = sz.x;
    int h = sz.y;

    // Compute texture-space u, v.
    float u = uv.x;
    float v = uv.y;
    bool clampU = false;
    bool clampV = false;

    // Cube map indexing.
    int face = 0;

    if (p.boundaryMode == TEX_BOUNDARY_MODE_WRAP)
    {
        // Wrap.
        u = u - (float)__float2int_rd(u);
        v = v - (float)__float2int_rd(v);
    }

    // Move to texel space.
    u = u * (float)w - 0.5f;
    v = v * (float)h - 0.5f;

    if (p.boundaryMode == TEX_BOUNDARY_MODE_CLAMP)
    {
        // Clamp to center of edge texels.
        u = fminf(fmaxf(u, 0.f), w - 1.f);
        v = fminf(fmaxf(v, 0.f), h - 1.f);
        clampU = (u == 0.f || u == w - 1.f);
        clampV = (v == 0.f || v == h - 1.f);
    }

    // Compute texel coordinates and weights.
    int iu0 = __float2int_rd(u);
    int iv0 = __float2int_rd(v);
    int iu1 = iu0 + (clampU ? 0 : 1); // Ensure zero u/v gradients with clamped.
    int iv1 = iv0 + (clampV ? 0 : 1);
    u -= (float)iu0;
    v -= (float)iv0;

    // Wrap overflowing texel indices.
    if (!CUBE_MODE && p.boundaryMode == TEX_BOUNDARY_MODE_WRAP)
    {
        if (iu0 < 0) iu0 += w;
        if (iv0 < 0) iv0 += h;
        if (iu1 >= w) iu1 -= w;
        if (iv1 >= h) iv1 -= h;
    }

    // Coordinates with tz folded in.
    int iu0z = iu0 + tz * w * h;
    int iu1z = iu1 + tz * w * h;
    tcOut.x = iu0z + w * iv0;
    tcOut.y = iu1z + w * iv0;
    tcOut.z = iu0z + w * iv1;
    tcOut.w = iu1z + w * iv1;

    // Invalidate texture addresses outside unit square if we are in zero mode.
    if (!CUBE_MODE && p.boundaryMode == TEX_BOUNDARY_MODE_ZERO)
    {
        bool iu0_out = (iu0 < 0 || iu0 >= w);
        bool iu1_out = (iu1 < 0 || iu1 >= w);
        bool iv0_out = (iv0 < 0 || iv0 >= h);
        bool iv1_out = (iv1 < 0 || iv1 >= h);
        if (iu0_out || iv0_out) tcOut.x = -1;
        if (iu1_out || iv0_out) tcOut.y = -1;
        if (iu0_out || iv1_out) tcOut.z = -1;
        if (iu1_out || iv1_out) tcOut.w = -1;
    }

    // All done.
    return make_float2(u, v);
}

//------------------------------------------------------------------------
// Texel fetch and accumulator helpers that understand cube map corners.

template<class T>
static __device__ __forceinline__ void fetchQuad(T& a00, T& a10, T& a01, T& a11, const float* pIn, int4 tc, bool corner)
{
    // For invalid cube map uv, tc will be all negative, and all texel values will be zero.
    if (corner)
    {
        T avg = zero_value<T>();
        if (tc.x >= 0) avg += (a00 = *((const T*)&pIn[tc.x]));
        if (tc.y >= 0) avg += (a10 = *((const T*)&pIn[tc.y]));
        if (tc.z >= 0) avg += (a01 = *((const T*)&pIn[tc.z]));
        if (tc.w >= 0) avg += (a11 = *((const T*)&pIn[tc.w]));
        avg *= 0.33333333f;
        if (tc.x < 0) a00 = avg;
        if (tc.y < 0) a10 = avg;
        if (tc.z < 0) a01 = avg;
        if (tc.w < 0) a11 = avg;
    }
    else
    {
        a00 = (tc.x >= 0) ? *((const T*)&pIn[tc.x]) : zero_value<T>();
        a10 = (tc.y >= 0) ? *((const T*)&pIn[tc.y]) : zero_value<T>();
        a01 = (tc.z >= 0) ? *((const T*)&pIn[tc.z]) : zero_value<T>();
        a11 = (tc.w >= 0) ? *((const T*)&pIn[tc.w]) : zero_value<T>();
    }
}


//------------------------------------------------------------------------
// Forward kernel.

template <class T, int C, bool CUBE_MODE, bool BIAS_ONLY, int FILTER_MODE>
static __forceinline__ __device__ void TextureFwdKernelTemplate(const TextureKernelParams p)
{
    // Calculate pixel position.
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    int tz = (p.texDepth == 1) ? 0 : pz;
    if (px >= p.imgWidth || py >= p.imgHeight || pz >= p.n)
        return;

    // Pixel index.
    int pidx = px + p.imgWidth * (py + p.imgHeight * pz);

    // Output ptr.
    float* pOut = p.out + pidx * p.channels;

    // Get UV.
    float3 uv;
    uv = make_float3(((const float2*)p.uv)[pidx], 0.f);

    // Calculate mip level. In 'linear' mode these will all stay zero.
    float  flevel = 0.f; // Fractional level.
    int    level0 = 0;   // Discrete level 0.
    int    level1 = 0;   // Discrete level 1.

    // Get texel indices and pointer for level 0.
    int4 tc0 = make_int4(0, 0, 0, 0);
    float2 uv0 = indexTextureLinear<CUBE_MODE>(p, uv, tz, tc0, level0);
    const float* pIn0 = p.tex[level0];
    bool corner0 = CUBE_MODE && ((tc0.x | tc0.y | tc0.z | tc0.w) < 0);
    tc0 *= p.channels;

    // Bilinear fetch.
    if (FILTER_MODE == TEX_MODE_LINEAR || FILTER_MODE == TEX_MODE_LINEAR_MIPMAP_NEAREST)
    {
        // Interpolate.
        for (int i=0; i < p.channels; i += C, tc0 += C)
        {
            T a00, a10, a01, a11;
            fetchQuad<T>(a00, a10, a01, a11, pIn0, tc0, corner0);
            *((T*)&pOut[i]) = bilerp(a00, a10, a01, a11, uv0);
        }
        return; // Exit.
    }
}

// Template specializations.
__global__ void TextureFwdKernelLinear1                     (const TextureKernelParams p) { TextureFwdKernelTemplate<float,  1, false, false, TEX_MODE_LINEAR>(p); }
//------------------------------------------------------------------------
