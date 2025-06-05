# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.22.1)
project(gxf_isaac_foundationpose LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-fPIC -w)
endif()

enable_language(CUDA)
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Dependencies
find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(nvcv_types REQUIRED)
find_package(cvcuda REQUIRED)
find_package(assimp REQUIRED)
find_package(isaac_ros_nitros_detection3_d_array_type REQUIRED)
find_package(yaml-cpp)
find_package(OpenCV REQUIRED)

# FoundationPose extension
ament_auto_add_library(${PROJECT_NAME} SHARED
    gxf/foundationpose/nvdiffrast/common/cudaraster/impl/Buffer.cpp
    gxf/foundationpose/nvdiffrast/common/cudaraster/impl/CudaRaster.cpp
    gxf/foundationpose/nvdiffrast/common/cudaraster/impl/RasterImpl.cpp
    gxf/foundationpose/nvdiffrast/common/cudaraster/impl/RasterImpl.cu
    gxf/foundationpose/nvdiffrast/common/common.cpp
    gxf/foundationpose/nvdiffrast/common/interpolate.cu
    gxf/foundationpose/nvdiffrast/common/rasterize.cu
    gxf/foundationpose/nvdiffrast/common/texture.cu
    
    gxf/foundationpose/foundationpose_sampling.cpp
    gxf/foundationpose/foundationpose_sampling.cu
    gxf/foundationpose/foundationpose_sync.cpp
    gxf/foundationpose/foundationpose_render.cpp
    gxf/foundationpose/foundationpose_render.cu
    gxf/foundationpose/foundationpose_decoder.cpp
    gxf/foundationpose/foundationpose_decoder.cu
    gxf/foundationpose/foundationpose_transformation.cpp
    gxf/foundationpose/mesh_storage.cpp
    gxf/foundationpose/foundationpose_ext.cpp
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    ${isaac_ros_nitros_detection3_d_array_type_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_compile_options(${PROJECT_NAME} PUBLIC -fPIC)

target_link_libraries(${PROJECT_NAME}
    CUDA::cudart
    Eigen3::Eigen
    yaml-cpp
    cvcuda
    assimp
    nvcv_types
    ${OpenCV_LIBS}
    ${isaac_ros_nitros_detection3_d_array_type_LIBRARIES}
)

set_target_properties(${PROJECT_NAME} PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Install the binary file
install(TARGETS ${PROJECT_NAME} DESTINATION share/${PROJECT_NAME}/gxf/lib)


# Embed versioning information into installed files
ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})

ament_auto_package(INSTALL_TO_SHARE)