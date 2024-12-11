// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gmock/gmock.h>
#include "centerpose_decoder_node.hpp"
#include "rclcpp/rclcpp.hpp"

// Objective: to cover code lines where exceptions are thrown
// Approach: send Invalid Arguments for node parameters to trigger the exception


TEST(centerpose_decoder_node_test, test_invalid_output_field_size)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::centerpose::CenterPoseDecoderNode centerpose_decoder_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("Error: received invalid output field size"));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}

TEST(centerpose_decoder_node_test, test_cuboid_scaling_factor)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("output_field_size", std::vector<int64_t>{128, 128});
  options.append_parameter_override("cuboid_scaling_factor", 0.0);
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::centerpose::CenterPoseDecoderNode centerpose_decoder_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(
        e.what(),
        testing::HasSubstr("Error: received a less than or equal to zero cuboid scaling factor"));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}

TEST(centerpose_decoder_node_test, test_score_threshold)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("output_field_size", std::vector<int64_t>{128, 128});
  options.append_parameter_override("cuboid_scaling_factor", 1.0);
  options.append_parameter_override("score_threshold", 1.0);
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::centerpose::CenterPoseDecoderNode centerpose_decoder_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(
        e.what(),
        testing::HasSubstr("Error: received score threshold greater or equal to 1.0"));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}


int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
