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

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

namespace nvidia
{
namespace isaac_ros
{
namespace foundationpose
{

/*
ROS 2 node that generates a binary segmentation mask image from
1. a vision_msgs::msg::Detection2D or vision_msgs::msg::Detection2DArray
2. image_height and image_width read from ROS parameters
*/
class Detection2DToMask : public rclcpp::Node
{
public:
  explicit Detection2DToMask(const rclcpp::NodeOptions & options)
  : Node("detection2_d_to_mask", options)
  {
    // Create a publisher for the mono8 image
    image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("segmentation", 10);

    this->declare_parameter<int>("mask_width", 640);
    this->declare_parameter<int>("mask_height", 480);
    this->get_parameter("mask_width", mask_width_);
    this->get_parameter("mask_height", mask_height_);

    detection2_d_array_sub_ = this->create_subscription<vision_msgs::msg::Detection2DArray>(
      "detection2_d_array", 10,
      std::bind(&Detection2DToMask::boundingBoxArrayCallback, this, std::placeholders::_1));
    // Create subscriber for Detection2D
    detection2_d_sub_ = this->create_subscription<vision_msgs::msg::Detection2D>(
      "detection2_d", 10,
      std::bind(&Detection2DToMask::boundingBoxCallback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Mask Height: %d, Mask Width: %d", mask_height_, mask_width_);
  }

  void boundingBoxCallback(const vision_msgs::msg::Detection2D::SharedPtr msg)
  {
    // Convert Detection2D to a binary mono8 image
    cv::Mat image = cv::Mat::zeros(mask_height_, mask_width_, CV_8UC1);
    // Draws a rectangle filled with 255
    cv::rectangle(
      image,
      cv::Point(
        msg->bbox.center.position.x - msg->bbox.size_x / 2,
        msg->bbox.center.position.y - msg->bbox.size_y / 2),
      cv::Point(
        msg->bbox.center.position.x + msg->bbox.size_x / 2,
        msg->bbox.center.position.y + msg->bbox.size_y / 2),
      cv::Scalar(255), -1);

    // Convert the OpenCV image to a ROS sensor_msgs::msg::Image and publish it
    std_msgs::msg::Header header(msg->header);
    cv_bridge::CvImage cv_image(header, "mono8", image);
    sensor_msgs::msg::Image image_msg;
    cv_image.toImageMsg(image_msg);
    image_pub_->publish(image_msg);
  }

  void boundingBoxArrayCallback(const vision_msgs::msg::Detection2DArray::SharedPtr msg)
  {
    // Find the detection bounding box with the highest confidence
    float max_confidence = 0;
    vision_msgs::msg::Detection2D max_confidence_detection;
    // Iterate through the detections and find the one with the highest confidence
    for (auto detection : msg->detections) {
      // Iterate through all the hypotheses for this detection
      // and find the one with the highest confidence
      for (auto result : detection.results) {
        if (result.hypothesis.score > max_confidence) {
          max_confidence = result.hypothesis.score;
          max_confidence_detection = detection;
        }
      }
    }

    // If no detection was found, return error
    if (max_confidence == 0) {
      RCLCPP_INFO(this->get_logger(), "No detection found with non-zero confidence");
      return;
    }

    // Convert Detection2D to a binary mono8 image
    cv::Mat image = cv::Mat::zeros(mask_height_, mask_width_, CV_8UC1);
    // Draws a rectangle filled with 255
    cv::rectangle(
      image,
      cv::Point(
        max_confidence_detection.bbox.center.position.x - max_confidence_detection.bbox.size_x / 2,
        max_confidence_detection.bbox.center.position.y - max_confidence_detection.bbox.size_y / 2),
      cv::Point(
        max_confidence_detection.bbox.center.position.x + max_confidence_detection.bbox.size_x / 2,
        max_confidence_detection.bbox.center.position.y + max_confidence_detection.bbox.size_y / 2),
      cv::Scalar(255), -1);

    // Convert the OpenCV image to a ROS sensor_msgs::msg::Image and publish it
    std_msgs::msg::Header header(msg->header);
    cv_bridge::CvImage cv_image(header, "mono8", image);
    sensor_msgs::msg::Image image_msg;
    cv_image.toImageMsg(image_msg);
    image_pub_->publish(image_msg);
  }

private:
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Subscription<vision_msgs::msg::Detection2D>::SharedPtr detection2_d_sub_;
  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detection2_d_array_sub_;
  int mask_height_;
  int mask_width_;
};

}  // namespace foundationpose
}  // namespace isaac_ros
}  // namespace nvidia

// Register the component with the ROS system to create a shared library
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::foundationpose::Detection2DToMask)
