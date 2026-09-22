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

#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"

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
  : Node("detection2_d_to_mask", options),
    input_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")},
    output_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")},
    mask_width_(declare_parameter<int>("mask_width", 640)),
    mask_height_(declare_parameter<int>("mask_height", 480)),
    detection2_d_sub_{create_subscription<vision_msgs::msg::Detection2D>(
        "detection2_d", input_qos_,
        std::bind(&Detection2DToMask::boundingBoxCallback, this, std::placeholders::_1))}
  {
    CHECK_CUDA_ERROR(
      cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking),
      "Failed to create CUDA stream");
    rclcpp::PublisherOptions pub_options;
    pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
    image_pub_ = create_publisher<sensor_msgs::msg::Image>(
      "segmentation", output_qos_, pub_options);
    RCLCPP_INFO(this->get_logger(), "Mask Height: %d, Mask Width: %d", mask_height_, mask_width_);
  }

  ~Detection2DToMask() override
  {
    cudaStreamDestroy(cuda_stream_);
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

    sensor_msgs::msg::Image image_msg;
    image_msg.header = msg->header;
    image_msg.height = mask_height_;
    image_msg.width = mask_width_;
    image_msg.encoding = "mono8";
    image_msg.is_bigendian = false;
    image_msg.step = mask_width_;
    image_msg.data = cuda_buffer_backend::allocate_buffer(image.total());
    {
      auto output = cuda_buffer_backend::from_output_buffer(image_msg.data, cuda_stream_);
      CHECK_CUDA_ERROR(
        cudaMemcpyAsync(
          output.get_ptr(), image.data, image.total(), cudaMemcpyHostToDevice, cuda_stream_),
        "Failed to upload segmentation mask");
    }
    image_pub_->publish(image_msg);
  }

private:
  // QOS settings
  rclcpp::QoS input_qos_;
  rclcpp::QoS output_qos_;

  int mask_width_;
  int mask_height_;
  cudaStream_t cuda_stream_{nullptr};
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Subscription<vision_msgs::msg::Detection2D>::SharedPtr detection2_d_sub_;
};

}  // namespace foundationpose
}  // namespace isaac_ros
}  // namespace nvidia

// Register the component with the ROS system to create a shared library
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::foundationpose::Detection2DToMask)
