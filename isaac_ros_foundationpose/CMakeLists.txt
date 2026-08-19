# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(isaac_ros_foundationpose LANGUAGES C CXX CUDA)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(assimp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(nvcv_types REQUIRED)
find_package(cvcuda REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SwitchMesh.srv"
)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

set(FOUNDATIONPOSE_IMPL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/foundationpose_impl")
set(NVDIFFRAST_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/nvdiffrast")

set(CUDA_KERNEL_SOURCES
  ${FOUNDATIONPOSE_IMPL_DIR}/foundationpose_sampling.cu
  ${FOUNDATIONPOSE_IMPL_DIR}/foundationpose_render.cu
  ${NVDIFFRAST_SRC_DIR}/common/interpolate.cu
  ${NVDIFFRAST_SRC_DIR}/common/rasterize.cu
  ${NVDIFFRAST_SRC_DIR}/common/texture.cu
  ${NVDIFFRAST_SRC_DIR}/common/cudaraster/impl/RasterImpl.cu
)

set(NVDIFFRAST_CPP_SOURCES
  ${NVDIFFRAST_SRC_DIR}/common/common.cpp
  ${NVDIFFRAST_SRC_DIR}/common/cudaraster/impl/Buffer.cpp
  ${NVDIFFRAST_SRC_DIR}/common/cudaraster/impl/CudaRaster.cpp
  ${NVDIFFRAST_SRC_DIR}/common/cudaraster/impl/RasterImpl.cpp
)

ament_auto_add_library(foundationpose_pipeline SHARED
  src/foundationpose_impl/mesh_loader.cpp
  src/foundationpose_impl/pose_sampler.cpp
  src/foundationpose_impl/pose_renderer.cpp
  src/foundationpose_impl/pose_transformer.cpp
  src/foundationpose_impl/pose_decoder.cpp
  ${CUDA_KERNEL_SOURCES}
  ${NVDIFFRAST_CPP_SOURCES}
)
set_target_properties(foundationpose_pipeline PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(foundationpose_pipeline PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-w>)
target_compile_options(foundationpose_pipeline PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-w>)
target_include_directories(foundationpose_pipeline PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/isaac_ros_foundationpose/foundationpose_impl>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${FOUNDATIONPOSE_IMPL_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${NVDIFFRAST_SRC_DIR}/common
  ${NVDIFFRAST_SRC_DIR}/common/cudaraster
  ${NVDIFFRAST_SRC_DIR}/common/cudaraster/impl
)
target_link_libraries(foundationpose_pipeline
  CUDA::cudart
  Eigen3::Eigen
  assimp::assimp
  cvcuda
  nvcv_types
  ${OpenCV_LIBS}
)

set_target_properties(foundationpose_pipeline PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

# FoundationPose node
ament_auto_add_library(foundationpose_node SHARED src/foundationpose_node.cpp)
ament_target_dependencies(foundationpose_node rclcpp_components)
target_link_libraries(foundationpose_node foundationpose_pipeline)
target_link_libraries(foundationpose_node "${cpp_typesupport_target}")
rclcpp_components_register_nodes(foundationpose_node "nvidia::isaac_ros::foundationpose::FoundationPoseNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::foundationpose::FoundationPoseNode;$<TARGET_FILE:foundationpose_node>\n")

set_target_properties(foundationpose_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

# FoundationPose Tracking node
ament_auto_add_library(foundationpose_tracking_node SHARED src/foundationpose_tracking_node.cpp)
ament_target_dependencies(foundationpose_tracking_node rclcpp_components)
target_link_libraries(foundationpose_tracking_node foundationpose_pipeline)
target_link_libraries(foundationpose_tracking_node "${cpp_typesupport_target}")
rclcpp_components_register_nodes(foundationpose_tracking_node "nvidia::isaac_ros::foundationpose::FoundationPoseTrackingNode")
set(node_plugins
  "${node_plugins}nvidia::isaac_ros::foundationpose::FoundationPoseTrackingNode;$<TARGET_FILE:foundationpose_tracking_node>\n")

# Detection2DArrayFilter utils node
ament_auto_add_library(detection2_d_array_filter SHARED src/detection2_d_array_filter.cpp)
ament_target_dependencies(detection2_d_array_filter rclcpp_components)
rclcpp_components_register_nodes(detection2_d_array_filter "nvidia::isaac_ros::foundationpose::Detection2DArrayFilter")
set(node_plugins "${node_plugins}nvidia::isaac_ros::foundationpose::Detection2DArrayFilter;$<TARGET_FILE:detection2_d_array_filter>\n")

# Detection2DToMask utils node
ament_auto_add_library(detection2_d_to_mask SHARED src/detection2_d_to_mask.cpp)
ament_target_dependencies(detection2_d_to_mask rclcpp_components)
rclcpp_components_register_nodes(detection2_d_to_mask "nvidia::isaac_ros::foundationpose::Detection2DToMask")
set(node_plugins "${node_plugins}nvidia::isaac_ros::foundationpose::Detection2DToMask;$<TARGET_FILE:detection2_d_to_mask>\n")

# Tracking selector node
ament_auto_add_library(foundationpose_selector_node SHARED src/foundationpose_selector_node.cpp)
ament_target_dependencies(foundationpose_selector_node rclcpp_components)
rclcpp_components_register_nodes(foundationpose_selector_node "nvidia::isaac_ros::foundationpose::Selector")
set(node_plugins "${node_plugins}nvidia::isaac_ros::foundationpose::Selector;$<TARGET_FILE:foundationpose_selector_node>\n")

set_target_properties(foundationpose_selector_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  file(GLOB_RECURSE CUDA_SRC_FILES
    src/foundationpose_impl/*.cu
    src/foundationpose_impl/*.cu.hpp
    src/nvdiffrast/*)
  set(AMENT_LINT_AUTO_FILE_EXCLUDE
    ${CUDA_SRC_FILES}
  )
  ament_lint_auto_find_test_dependencies()

  # The FindPythonInterp and FindPythonLibs modules are removed
  if(POLICY CMP0148)
    cmake_policy(SET CMP0148 OLD)
  endif()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_foundationpose_pol.py TIMEOUT "900")
  add_launch_test(test/isaac_ros_foundationpose_bbox_pol.py TIMEOUT "900")
  add_launch_test(test/isaac_ros_foundationpose_fixed_axis_angles_pol.py TIMEOUT "900")
  add_launch_test(test/isaac_ros_foundationpose_symmetry_axes_pol.py TIMEOUT "900")
  add_launch_test(test/isaac_ros_foundationpose_iterative_refine_pol.py TIMEOUT "900")
  add_launch_test(test/isaac_ros_foundationpose_tracking_pol.py TIMEOUT "900")
endif()


# Embed versioning information into installed files
ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})

ament_auto_package(
  INSTALL_TO_SHARE config launch rviz
  USE_SCOPED_HEADER_INSTALL_DIR
)
