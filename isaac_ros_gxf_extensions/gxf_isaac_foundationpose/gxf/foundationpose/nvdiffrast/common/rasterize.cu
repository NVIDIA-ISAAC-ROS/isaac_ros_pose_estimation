// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "common.h"
#include "rasterize.h"

//------------------------------------------------------------------------
// Cuda forward rasterizer pixel shader kernel.

__global__ void RasterizeCudaFwdShaderKernel(const RasterizeCudaFwdShaderParams p)
{
    // Calculate pixel position.
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.width || py >= p.height || pz >= p.depth)
        return;

    // Pixel index.
    int pidx = px + p.width * (py + p.height * pz);

    // Fetch triangle idx.
    int triIdx = p.in_idx[pidx] - 1;
    if (triIdx < 0 || triIdx >= p.numTriangles)
    {
        // No or corrupt triangle.
        ((float4*)p.out)[pidx] = make_float4(0.0, 0.0, 0.0, 0.0); // Clear out.
        return;
    }

    // Fetch vertex indices.
    int vi0 = p.tri[triIdx * 3 + 0];
    int vi1 = p.tri[triIdx * 3 + 1];
    int vi2 = p.tri[triIdx * 3 + 2];

    // Bail out if vertex indices are corrupt.
    if (vi0 < 0 || vi0 >= p.numVertices ||
        vi1 < 0 || vi1 >= p.numVertices ||
        vi2 < 0 || vi2 >= p.numVertices)
        return;

    // In instance mode, adjust vertex indices by minibatch index.
    if (p.instance_mode)
    {
        vi0 += pz * p.numVertices;
        vi1 += pz * p.numVertices;
        vi2 += pz * p.numVertices;
    }

    // Fetch vertex positions.
    float4 p0 = ((float4*)p.pos)[vi0];
    float4 p1 = ((float4*)p.pos)[vi1];
    float4 p2 = ((float4*)p.pos)[vi2];

    // Evaluate edge functions.
    float fx = p.xs * (float)px + p.xo;
    float fy = p.ys * (float)py + p.yo;
    float p0x = p0.x - fx * p0.w;
    float p0y = p0.y - fy * p0.w;
    float p1x = p1.x - fx * p1.w;
    float p1y = p1.y - fy * p1.w;
    float p2x = p2.x - fx * p2.w;
    float p2y = p2.y - fy * p2.w;
    float a0 = p1x*p2y - p1y*p2x;
    float a1 = p2x*p0y - p2y*p0x;
    float a2 = p0x*p1y - p0y*p1x;

    // Perspective correct, normalized barycentrics.
    float iw = 1.f / (a0 + a1 + a2);
    float b0 = a0 * iw;
    float b1 = a1 * iw;

    // Compute z/w for depth buffer.
    float z = p0.z * a0 + p1.z * a1 + p2.z * a2;
    float w = p0.w * a0 + p1.w * a1 + p2.w * a2;
    float zw = z / w;

    // Clamps to avoid NaNs.
    b0 = __saturatef(b0); // Clamp to [+0.0, 1.0].
    b1 = __saturatef(b1); // Clamp to [+0.0, 1.0].
    zw = fmaxf(fminf(zw, 1.f), -1.f);

    // Emit output.
    ((float4*)p.out)[pidx] = make_float4(b0, b1, zw, (float)(triIdx + 1));
}

//------------------------------------------------------------------------
