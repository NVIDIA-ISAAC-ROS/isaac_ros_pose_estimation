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

#include "Defs.hpp"
#include "../CudaRaster.hpp"
#include "RasterImpl.hpp"

using namespace CR;

//------------------------------------------------------------------------
// Stub interface implementation.
//------------------------------------------------------------------------

CudaRaster::CudaRaster()
{
    m_impl = new RasterImpl();
}

CudaRaster::~CudaRaster()
{
    delete m_impl;
}

void CudaRaster::setViewportSize(int width, int height, int numImages)
{
    m_impl->setViewportSize(Vec3i(width, height, numImages));
}

void CudaRaster::setRenderModeFlags(U32 flags)
{
    m_impl->setRenderModeFlags(flags);
}

void CudaRaster::deferredClear(U32 clearColor)
{
    m_impl->deferredClear(clearColor);
}

void CudaRaster::setVertexBuffer(void* vertices, int numVertices)
{
    m_impl->setVertexBuffer(vertices, numVertices);
}

void CudaRaster::setIndexBuffer(void* indices, int numTriangles)
{
    m_impl->setIndexBuffer(indices, numTriangles);
}

bool CudaRaster::drawTriangles(const int* ranges, bool peel, cudaStream_t stream)
{
    return m_impl->drawTriangles((const Vec2i*)ranges, peel, stream);
}

void* CudaRaster::getColorBuffer(void)
{
    return m_impl->getColorBuffer();
}

void* CudaRaster::getDepthBuffer(void)
{
    return m_impl->getDepthBuffer();
}


//------------------------------------------------------------------------
