# Using DOPE with a non-standard input image size

## Overview

The DOPE network architecture, as outlined in the [original paper](https://arxiv.org/abs/1809.10790), can receive input images of arbitrary size and subsequently produce output belief maps of the corresponding dimensions.

However, the ONNX format used to run this network on Triton or TensorRT is not as flexible, and an ONNX-exported model **does NOT** support arbitrary image sizes at inference time. Instead, the desired input image dimensions must be explicitly specified when preparing the ONNX file using the `dope_converter.py` script, as referenced in the [quickstart](../README.md#quickstart).

## Tutorial Walkthrough

1. Follow steps 1-6 of the main DOPE [quickstart](../README.md#quickstart).

2. At step 7, run the `dope_converter.py` script with the two additional arguments `row` and `col` specifying the desired input image size:

    ```bash
    python3 /workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation/isaac_ros_dope/scripts/dope_converter.py --format onnx --input /tmp/models/Ketchup.pth --row 1080 --col 1920
    ```

3. Proceed through steps 8-9.
4. At step 10, launch the ROS 2 launchfile with two additional arguments `network_image_height` and `network_image_width` specifying the desired input image size:

    ```bash
    ros2 launch isaac_ros_dope isaac_ros_dope_tensor_rt.launch.py model_file_path:=/tmp/models/Ketchup.onnx engine_file_path:=/tmp/models/Ketchup.plan network_image_height:=1080 network_image_width:=1920
    ```

5. Continue with the rest of the quickstart. You should now be able to detect poses in images of your desired size.
