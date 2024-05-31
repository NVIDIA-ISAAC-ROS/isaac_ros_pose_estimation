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

#include "../../framework.h"
#include "Buffer.hpp"

using namespace CR;

//------------------------------------------------------------------------
// GPU buffer.
//------------------------------------------------------------------------

Buffer::Buffer(void)
:   m_gpuPtr(NULL),
    m_bytes (0)
{
    // empty
}

Buffer::~Buffer(void)
{
    if (m_gpuPtr)
        cudaFree(m_gpuPtr); // Don't throw an exception.
}

void Buffer::reset(size_t bytes)
{
    if (bytes == m_bytes)
        return;

    if (m_gpuPtr)
    {
        NVDR_CHECK_CUDA_ERROR(cudaFree(m_gpuPtr));
        m_gpuPtr = NULL;
    }

    if (bytes > 0)
        NVDR_CHECK_CUDA_ERROR(cudaMalloc(&m_gpuPtr, bytes));

    m_bytes = bytes;
}

void Buffer::grow(size_t bytes)
{
    if (bytes > m_bytes)
        reset(bytes);
}

//------------------------------------------------------------------------
// Host buffer with page-locked memory.
//------------------------------------------------------------------------

HostBuffer::HostBuffer(void)
:   m_hostPtr(NULL),
    m_bytes  (0)
{
    // empty
}

HostBuffer::~HostBuffer(void)
{
    if (m_hostPtr)
        cudaFreeHost(m_hostPtr); // Don't throw an exception.
}

void HostBuffer::reset(size_t bytes)
{
    if (bytes == m_bytes)
        return;

    if (m_hostPtr)
    {
        NVDR_CHECK_CUDA_ERROR(cudaFreeHost(m_hostPtr));
        m_hostPtr = NULL;
    }

    if (bytes > 0)
        NVDR_CHECK_CUDA_ERROR(cudaMallocHost(&m_hostPtr, bytes));

    m_bytes = bytes;
}

void HostBuffer::grow(size_t bytes)
{
    if (bytes > m_bytes)
        reset(bytes);
}

//------------------------------------------------------------------------
