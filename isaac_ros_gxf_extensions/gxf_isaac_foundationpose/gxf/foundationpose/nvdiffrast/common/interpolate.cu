// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "common.h"
#include "interpolate.h"

//------------------------------------------------------------------------
// Forward kernel.

template <bool ENABLE_DA>
static __forceinline__ __device__ void InterpolateFwdKernelTemplate(const InterpolateKernelParams p)
{
    // Calculate pixel position.
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.width || py >= p.height || pz >= p.depth)
        return;

    // Pixel index.
    int pidx = px + p.width * (py + p.height * pz);

    // Output ptrs.
    float* out = p.out + pidx * p.numAttr;

    // Fetch rasterizer output.
    float4 r = ((float4*)p.rast)[pidx];
    int triIdx = (int)r.w - 1;
    bool triValid = (triIdx >= 0 && triIdx < p.numTriangles);

    // If no geometry in entire warp, zero the output and exit.
    // Otherwise force barys to zero and output with live threads.
    if (__all_sync(0xffffffffu, !triValid))
    {
        for (int i=0; i < p.numAttr; i++)
            out[i] = 0.f;
        return;
    }

    // Fetch vertex indices.
    int vi0 = triValid ? p.tri[triIdx * 3 + 0] : 0;
    int vi1 = triValid ? p.tri[triIdx * 3 + 1] : 0;
    int vi2 = triValid ? p.tri[triIdx * 3 + 2] : 0;

    // Bail out if corrupt indices.
    if (vi0 < 0 || vi0 >= p.numVertices ||
        vi1 < 0 || vi1 >= p.numVertices ||
        vi2 < 0 || vi2 >= p.numVertices)
        return;

    // In instance mode, adjust vertex indices by minibatch index unless broadcasting.
    if (p.instance_mode && !p.attrBC)
    {
        vi0 += pz * p.numVertices;
        vi1 += pz * p.numVertices;
        vi2 += pz * p.numVertices;
    }

    // Pointers to attributes.
    const float* a0 = p.attr + vi0 * p.numAttr;
    const float* a1 = p.attr + vi1 * p.numAttr;
    const float* a2 = p.attr + vi2 * p.numAttr;

    // Barys. If no triangle, force all to zero -> output is zero.
    float b0 = triValid ? r.x : 0.f;
    float b1 = triValid ? r.y : 0.f;
    float b2 = triValid ? (1.f - r.x - r.y) : 0.f;

    // Interpolate and write attributes.
    for (int i=0; i < p.numAttr; i++)
        out[i] = b0*a0[i] + b1*a1[i] + b2*a2[i];

    // No diff attrs? Exit.
    if (!ENABLE_DA)
        return;
}

// Template specializations.
__global__ void InterpolateFwdKernel  (const InterpolateKernelParams p) { InterpolateFwdKernelTemplate<false>(p); }

//------------------------------------------------------------------------
