# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(gxf_isaac_centerpose LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-fPIC -w)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Dependencies
find_package(CUDAToolkit REQUIRED)
find_package (Eigen3 3.3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp)
find_package(isaac_ros_nitros_detection3_d_array_type REQUIRED)

# Centerpose extension
ament_auto_add_library(${PROJECT_NAME} SHARED
  gxf/extensions/centerpose/components/centerpose_postprocessor.cpp
  gxf/extensions/centerpose/components/centerpose_postprocessor.hpp
  gxf/extensions/centerpose/components/cuboid3d.hpp
  gxf/extensions/centerpose/components/cuboid3d.cpp
  gxf/extensions/centerpose/components/cuboid_pnp_solver.cpp
  gxf/extensions/centerpose/components/cuboid_pnp_solver.hpp
  gxf/extensions/centerpose/components/centerpose_detection.hpp
  gxf/extensions/centerpose/components/centerpose_detection_to_isaac.hpp
  gxf/extensions/centerpose/components/centerpose_detection_to_isaac.cpp
  gxf/extensions/centerpose/components/soft_nms_nvidia.cpp
  gxf/extensions/centerpose/components/soft_nms_nvidia.hpp
  gxf/extensions/centerpose/components/centerpose_visualizer.cpp
  gxf/extensions/centerpose/components/centerpose_visualizer.hpp
  gxf/extensions/centerpose/components/video_buffer_utils.hpp
  gxf/extensions/centerpose/centerpose_ext.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/gxf
  ${CMAKE_CURRENT_SOURCE_DIR}/gxf/extensions/centerpose/components
  ${OpenCV_INCLUDE_DIRS}
  ${isaac_ros_nitros_detection3_d_array_type_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
  CUDA::cudart
  Eigen3::Eigen
  ${OpenCV_LIBS}
  yaml-cpp
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
