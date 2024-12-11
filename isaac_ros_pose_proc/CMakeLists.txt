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
project(isaac_ros_pose_proc C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# AveragingFilterNode
ament_auto_add_library(averaging_filter_node SHARED src/averaging_filter_node.cpp)
rclcpp_components_register_nodes(averaging_filter_node "nvidia::isaac_ros::pose_proc::AveragingFilterNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::pose_proc::AveragingFilterNode;$<TARGET_FILE:averaging_filter_node>\n")
set_target_properties(averaging_filter_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# Detection3DArrayToPoseNode
ament_auto_add_library(detection3_d_array_to_pose_node SHARED src/detection3_d_array_to_pose_node.cpp)
rclcpp_components_register_nodes(detection3_d_array_to_pose_node "nvidia::isaac_ros::pose_proc::Detection3DArrayToPoseNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::pose_proc::Detection3DArrayToPoseNode;$<TARGET_FILE:detection3_d_array_to_pose_node>\n")
set_target_properties(detection3_d_array_to_pose_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# DistanceSelectorNode
ament_auto_add_library(distance_selector_node SHARED src/distance_selector_node.cpp)
rclcpp_components_register_nodes(distance_selector_node "nvidia::isaac_ros::pose_proc::DistanceSelectorNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::pose_proc::DistanceSelectorNode;$<TARGET_FILE:distance_selector_node>\n")
set_target_properties(distance_selector_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# OutlierFilterNode
ament_auto_add_library(outlier_filter_node SHARED src/outlier_filter_node.cpp)
rclcpp_components_register_nodes(outlier_filter_node "nvidia::isaac_ros::pose_proc::OutlierFilterNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::pose_proc::OutlierFilterNode;$<TARGET_FILE:outlier_filter_node>\n")
set_target_properties(outlier_filter_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# PoseArrayToPoseNode
ament_auto_add_library(pose_array_to_pose_node SHARED src/pose_array_to_pose_node.cpp)
rclcpp_components_register_nodes(pose_array_to_pose_node "nvidia::isaac_ros::pose_proc::PoseArrayToPoseNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::pose_proc::PoseArrayToPoseNode;$<TARGET_FILE:pose_array_to_pose_node>\n")
set_target_properties(pose_array_to_pose_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# StabilityFilterNode
ament_auto_add_library(stability_filter_node SHARED src/stability_filter_node.cpp)
rclcpp_components_register_nodes(stability_filter_node "nvidia::isaac_ros::pose_proc::StabilityFilterNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::pose_proc::StabilityFilterNode;$<TARGET_FILE:stability_filter_node>\n")
set_target_properties(stability_filter_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# Install python scripts
install(PROGRAMS
  scripts/pose_distribution_visualizer.py
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  # The FindPythonInterp and FindPythonLibs modules are removed
  if(POLICY CMP0148)
    cmake_policy(SET CMP0148 OLD)
  endif()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_averaging_filter_test.py)
  add_launch_test(test/isaac_ros_distance_selector_test.py)
  add_launch_test(test/isaac_ros_outlier_filter_test.py)
  add_launch_test(test/isaac_ros_stability_filter_test.py)
endif()


# Embed versioning information into installed files
ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})

ament_auto_package()
