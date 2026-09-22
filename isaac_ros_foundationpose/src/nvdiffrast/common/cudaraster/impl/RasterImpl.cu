// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2009-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../CudaRaster.hpp"
#include "PrivateDefs.hpp"
#include "Constants.hpp"
#include "Util.inl"

namespace CR
{

//------------------------------------------------------------------------
// Stage implementations.
//------------------------------------------------------------------------

#include "TriangleSetup.inl"
#include "BinRaster.inl"
#include "CoarseRaster.inl"
#include "FineRaster.inl"

}

//------------------------------------------------------------------------
// Stage entry points.
//------------------------------------------------------------------------

__global__ void __launch_bounds__(CR_SETUP_WARPS * 32, CR_SETUP_OPT_BLOCKS)  triangleSetupKernel (const CR::CRParams p)  { CR::triangleSetupImpl(p); }
__global__ void __launch_bounds__(CR_BIN_WARPS * 32, 1)                      binRasterKernel     (const CR::CRParams p)  { CR::binRasterImpl(p); }
__global__ void __launch_bounds__(CR_COARSE_WARPS * 32, 1)                   coarseRasterKernel  (const CR::CRParams p)  { CR::coarseRasterImpl(p); }
__global__ void __launch_bounds__(CR_FINE_MAX_WARPS * 32, 1)                 fineRasterKernel    (const CR::CRParams p)  { CR::fineRasterImpl(p); }

//------------------------------------------------------------------------
