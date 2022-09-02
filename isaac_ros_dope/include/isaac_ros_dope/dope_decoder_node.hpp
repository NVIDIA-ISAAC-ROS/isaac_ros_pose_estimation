/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#ifndef ISAAC_ROS_DOPE__DOPE_DECODER_NODE_HPP_
#define ISAAC_ROS_DOPE__DOPE_DECODER_NODE_HPP_

#include <string>
#include <vector>

#include "geometry_msgs/msg/pose_array.hpp"
#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"
#include "isaac_ros_nitros/nitros_node.hpp"
#include "rclcpp/rclcpp.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dope
{

/**
 * @class DopeDecoderNode
 * @brief This node performs pose estimation of a known object from a single RGB
 *        image
 *        Paper: See https://arxiv.org/abs/1809.10790
 *        Code: https://github.com/NVlabs/Deep_Object_Pose
 */
class DopeDecoderNode : public nitros::NitrosNode
{
public:
  explicit DopeDecoderNode(rclcpp::NodeOptions options = rclcpp::NodeOptions());
  ~DopeDecoderNode();

  void postLoadGraphCallback() override;

private:
  // The name of the YAML configuration file
  const std::string configuration_file_;

  // The class name of the object we're locating
  const std::string object_name_;

  // The dimensions of the cuboid around the object we're locating
  std::vector<double> object_dimensions_;

  // The camera matrix used to capture the input images
  std::vector<double> camera_matrix_;
};

}  // namespace dope
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_DOPE__DOPE_DECODER_NODE_HPP_
