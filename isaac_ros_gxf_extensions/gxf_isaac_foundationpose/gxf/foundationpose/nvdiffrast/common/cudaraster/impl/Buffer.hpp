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
#include "Defs.hpp"

namespace CR
{
//------------------------------------------------------------------------

class Buffer
{
public:
                    Buffer      (void);
                    ~Buffer     (void);

    void            reset       (size_t bytes);
    void            grow        (size_t bytes);
    void*           getPtr      (void) { return m_gpuPtr; }
    size_t          getSize     (void) const { return m_bytes; }

    void            setPtr      (void* ptr) { m_gpuPtr = ptr; }

private:
    void*           m_gpuPtr;
    size_t          m_bytes;
};

//------------------------------------------------------------------------

class HostBuffer
{
public:
                    HostBuffer  (void);
                    ~HostBuffer (void);

    void            reset       (size_t bytes);
    void            grow        (size_t bytes);
    void*           getPtr      (void) { return m_hostPtr; }
    size_t          getSize     (void) const { return m_bytes; }

    void            setPtr      (void* ptr) { m_hostPtr = ptr; }

private:
    void*           m_hostPtr;
    size_t          m_bytes;
};

//------------------------------------------------------------------------
}
