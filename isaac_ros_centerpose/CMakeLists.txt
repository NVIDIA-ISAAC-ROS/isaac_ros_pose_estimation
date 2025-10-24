# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(isaac_ros_centerpose LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Decoder node
ament_auto_add_library(centerpose_decoder_node SHARED src/centerpose_decoder_node.cpp)
rclcpp_components_register_nodes(centerpose_decoder_node "nvidia::isaac_ros::centerpose::CenterPoseDecoderNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::centerpose::CenterPoseDecoderNode;$<TARGET_FILE:centerpose_decoder_node>\n")
set_target_properties(centerpose_decoder_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Visualizer node
ament_auto_add_library(centerpose_visualizer_node SHARED src/centerpose_visualizer_node.cpp)
rclcpp_components_register_nodes(centerpose_visualizer_node "nvidia::isaac_ros::centerpose::CenterPoseVisualizerNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::centerpose::CenterPoseVisualizerNode;$<TARGET_FILE:centerpose_visualizer_node>\n")
set_target_properties(centerpose_visualizer_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  ament_lint_auto_find_test_dependencies()

  # Force use of ROS2 vendor googletest to avoid version conflicts
  set(CMAKE_IGNORE_PATH "/usr/src/googletest" ${CMAKE_IGNORE_PATH})
  set(CMAKE_IGNORE_PATH "/usr/include/gtest" ${CMAKE_IGNORE_PATH})
  set(CMAKE_IGNORE_PATH "/usr/include/gmock" ${CMAKE_IGNORE_PATH})

  # Explicitly use ROS2 vendor googletest/gmock
  include_directories(BEFORE /opt/ros/$ENV{ROS_DISTRO}/src/gtest_vendor/include)
  include_directories(BEFORE /opt/ros/$ENV{ROS_DISTRO}/src/gmock_vendor/include)

  # Gtest for decoder_node
  ament_add_gtest(centerpose_decoder_node_test test/centerpose_decoder_node_test.cpp)
  target_link_libraries(centerpose_decoder_node_test centerpose_decoder_node)
  target_include_directories(centerpose_decoder_node_test PUBLIC include/isaac_ros_centerpose/)
  ament_target_dependencies(centerpose_decoder_node_test rclcpp)
  ament_target_dependencies(centerpose_decoder_node_test isaac_ros_nitros)
  # The FindPythonInterp and FindPythonLibs modules are removed
  if(POLICY CMP0148)
    cmake_policy(SET CMP0148 OLD)
  endif()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/test_centerpose_pol.py TIMEOUT "600")
  add_launch_test(test/test_centerpose_pol_triton.py TIMEOUT "600")
endif()


# Embed versioning information into installed files
ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})

ament_auto_package(INSTALL_TO_SHARE config launch)
