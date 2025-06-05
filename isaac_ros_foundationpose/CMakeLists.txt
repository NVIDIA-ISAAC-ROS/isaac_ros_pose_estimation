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
project(isaac_ros_foundationpose LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# FoundationPose node
ament_auto_add_library(foundationpose_node SHARED src/foundationpose_node.cpp)
rclcpp_components_register_nodes(foundationpose_node "nvidia::isaac_ros::foundationpose::FoundationPoseNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::foundationpose::FoundationPoseNode;$<TARGET_FILE:foundationpose_node>\n")

# FoundationPose Tracking node
ament_auto_add_library(foundationpose_tracking_node SHARED src/foundationpose_tracking_node.cpp)
rclcpp_components_register_nodes(foundationpose_tracking_node "nvidia::isaac_ros::foundationpose::FoundationPoseTrackingNode")
set(node_plugins
  "${node_plugins}nvidia::isaac_ros::foundationpose::FoundationPoseTrackingNode;$<TARGET_FILE:foundationpose_tracking_node>\n")

# Detection2DArrayFilter utils node
ament_auto_add_library(detection2_d_array_filter SHARED src/detection2_d_array_filter.cpp)
rclcpp_components_register_nodes(detection2_d_array_filter "nvidia::isaac_ros::foundationpose::Detection2DArrayFilter")
set(node_plugins "${node_plugins}nvidia::isaac_ros::foundationpose::Detection2DArrayFilter;$<TARGET_FILE:detection2_d_array_filter>\n")

# Detection2DToMask utils node
ament_auto_add_library(detection2_d_to_mask SHARED src/detection2_d_to_mask.cpp)
rclcpp_components_register_nodes(detection2_d_to_mask "nvidia::isaac_ros::foundationpose::Detection2DToMask")
set(node_plugins "${node_plugins}nvidia::isaac_ros::foundationpose::Detection2DToMask;$<TARGET_FILE:detection2_d_to_mask>\n")

# Tracking selector node
ament_auto_add_library(foundationpose_selector_node SHARED src/foundationpose_selector_node.cpp)
rclcpp_components_register_nodes(foundationpose_selector_node "nvidia::isaac_ros::foundationpose::Selector")
set(node_plugins "${node_plugins}nvidia::isaac_ros::foundationpose::Selector;$<TARGET_FILE:foundationpose_selector_node>\n")

set_target_properties(foundationpose_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

set_target_properties(foundationpose_selector_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(AMENT_LINT_AUTO_FILE_EXCLUDE scripts/models/network_modules.py)
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

ament_auto_package(INSTALL_TO_SHARE config launch rviz)
