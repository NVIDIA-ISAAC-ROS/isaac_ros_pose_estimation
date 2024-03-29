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

project(gxf_centerpose LANGUAGES C CXX)

# Dependencies
find_package(CUDAToolkit REQUIRED)
find_package (Eigen3 3.3 REQUIRED)
find_package(GXF ${ISAAC_ROS_GXF_VERSION} MODULE REQUIRED
    COMPONENTS
    cuda
    multimedia
    std
    isaac_messages
)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp)
find_package(isaac_ros_nitros_detection3_d_array_type REQUIRED)

# Centerpose extension
add_library(gxf_centerpose SHARED
    components/centerpose_postprocessor.cpp
    components/centerpose_postprocessor.hpp
    components/cuboid3d.hpp
    components/cuboid3d.cpp
    components/cuboid_pnp_solver.cpp
    components/cuboid_pnp_solver.hpp 
    components/centerpose_detection.hpp
    components/centerpose_detection_to_isaac.hpp
    components/centerpose_detection_to_isaac.cpp
    components/soft_nms_nvidia.cpp
    components/soft_nms_nvidia.hpp
    components/centerpose_visualizer.cpp
    components/centerpose_visualizer.hpp
    components/video_buffer_utils.hpp
    centerpose_ext.cpp
)
target_include_directories(gxf_centerpose
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${isaac_ros_nitros_detection3_d_array_type_INCLUDE_DIRS}
)
target_link_libraries(gxf_centerpose
  PUBLIC
    CUDA::cudart
    Eigen3::Eigen
    GXF::cuda
    GXF::isaac_messages
    GXF::multimedia
    GXF::std
    ${OpenCV_LIBS}
    yaml-cpp
    ${isaac_ros_nitros_detection3_d_array_type_LIBRARIES}
)
