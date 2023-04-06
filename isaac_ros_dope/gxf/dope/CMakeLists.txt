# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

project(gxf_dope LANGUAGES C CXX)

# Dependencies
find_package(CUDAToolkit REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(GXF ${ISAAC_ROS_GXF_VERSION} MODULE REQUIRED
    COMPONENTS
    cuda
    multimedia
    std
)
find_package(OpenCV REQUIRED)
include(YamlCpp)

# DOPE extension
add_library(gxf_dope SHARED
    dope_decoder.cpp
    dope_decoder.hpp
    dope_ext.cpp
)
target_include_directories(gxf_dope PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(gxf_dope
  PUBLIC
    CUDA::cudart
    Eigen3::Eigen
    GXF::cuda
    GXF::multimedia
    GXF::std
    ${OpenCV_LIBS}
    yaml-cpp
)