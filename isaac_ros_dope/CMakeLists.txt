# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

cmake_minimum_required(VERSION 3.5)
project(isaac_ros_dope LANGUAGES C CXX)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Default to Release build
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
message( STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}" )

execute_process(COMMAND uname -m COMMAND tr -d '\n'
  OUTPUT_VARIABLE ARCHITECTURE
)
message( STATUS "Architecture: ${ARCHITECTURE}" )

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Decoder node
ament_auto_add_library(dope_decoder_node SHARED
  src/dope_decoder_node.cpp)
target_compile_definitions(dope_decoder_node
  PRIVATE "COMPOSITION_BUILDING_DLL"
)
rclcpp_components_register_nodes(dope_decoder_node
  "nvidia::isaac_ros::dope::DopeDecoderNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::dope::DopeDecoderNode;$<TARGET_FILE:dope_decoder_node>\n")

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS dope_decoder_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install extensions directory
if( ${ARCHITECTURE} STREQUAL "x86_64" )
  install(DIRECTORY gxf/lib/gxf_x86_64/ DESTINATION share/${PROJECT_NAME}/gxf)
elseif( ${ARCHITECTURE} STREQUAL "aarch64" )
  install(DIRECTORY gxf/lib/gxf_jetpack50/ DESTINATION share/${PROJECT_NAME}/gxf)
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  # Ignore copyright notices since we use custom NVIDIA Isaac ROS Software License
  set(ament_cmake_copyright_FOUND TRUE)

  ament_lint_auto_find_test_dependencies()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_dope_pol.py TIMEOUT "600")
endif()

ament_auto_package(INSTALL_TO_SHARE launch)
