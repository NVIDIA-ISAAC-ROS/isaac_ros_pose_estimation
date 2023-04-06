# Tutorial for DOPE Inference

<div align="center"><img src="../resources/dope_rviz2.png" width="600px"/></div>

## Overview

This tutorial walks you through a graph to estimate the 6DOF pose of a target object using [DOPE](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation) using different backends. It uses input monocular images from a rosbag. The different backends show are:

1. PyTorch and ONNX
2. TensorRT Plan files with Triton
3. PyTorch model with Triton

> **Note**: The DOPE converter script only works on `x86_64`, so the resultant `onnx` model following these steps must be copied to the Jetson.

## Tutorial Walkthrough

1. Complete steps 1-6 of the quickstart [here](../README.md#quickstart).
2. Make a directory called `Ketchup` inside `/tmp/models`, which will serve as the model repository. This will be versioned as `1`. The downloaded model will be placed here:

    ```bash
    mkdir -p /tmp/models/Ketchup/1 && \
      mv /tmp/models/Ketchup.pth /tmp/models/Ketchup/
    ```

3. Now select a backend. The PyTorch and ONNX options **MUST** be run on `x86_64`:
   - To run ONNX models with Triton, export the model into an ONNX file using the script provided under `/workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation/isaac_ros_dope/scripts/dope_converter.py`:

     ```bash
     python3 /workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation/isaac_ros_dope/scripts/dope_converter.py --format onnx --input /tmp/models/Ketchup/Ketchup.pth --output /tmp/models/Ketchup/1/model.onnx --input_name INPUT__0 --output_name OUTPUT__0
     ```

   - To run `TensorRT Plan` files with Triton, first copy the generated `onnx` model from the above point to the target platform (e.g. a Jetson or an `x86_64` machine). The model will be assumed to be copied to `/tmp/models/Ketchup/1/model.onnx` inside the Docker container. Then use `trtexec` to convert the `onnx` model to a `plan` model:

     ```bash
     /usr/src/tensorrt/bin/trtexec --onnx=/tmp/models/Ketchup/1/model.onnx --saveEngine=/tmp/models/Ketchup/1/model.plan
     ```

   - To run PyTorch model with Triton (**inferencing PyTorch model is supported for x86_64 platform only**), the model needs to be saved using `torch.jit.save()`. The downloaded DOPE model is saved with `torch.save()`. Export the DOPE model using the script provided under `/workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation/isaac_ros_dope/scripts/dope_converter.py`:

     ```bash
     python3 /workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation/isaac_ros_dope/scripts/dope_converter.py --format pytorch --input /tmp/models/Ketchup/Ketchup.pth --output /tmp/models/Ketchup/1/model.pt
     ```

4. Create a configuration file for this model at path `/tmp/models/Ketchup/config.pbtxt`. Note that name has to be the same as the model repository. Depending on the platform selected from a previous step, a slightly different `config.pbtxt` file must be created: `onnxruntime_onnx` (`.onnx` file), `tensorrt_plan` (`.plan` file) or `pytorch_libtorch` (`.pt` file):

    ```log
    name: "Ketchup"
    platform: <insert-platform>
    max_batch_size: 0
    input [
      {
        name: "INPUT__0"
        data_type: TYPE_FP32
        dims: [ 1, 3, 480, 640 ]
      }
    ]
    output [
      {
        name: "OUTPUT__0"
        data_type: TYPE_FP32
        dims: [ 1, 25, 60, 80 ]
      }
    ]
    version_policy: {
      specific {
        versions: [ 1 ]
      }
    }
    ```

    The `<insert-platform>` part should be replaced with `onnxruntime_onnx` for `.onnx` files, `tensorrt_plan` for `.plan` files and `pytorch_libtorch` for `.pt` files.

    > **Note**: The DOPE decoder currently works with the output of a DOPE network that has a fixed input size of 640 x 480, which are the default dimensions set in the script. In order to use input images of other sizes, make sure to crop or resize using ROS 2 nodes from [Isaac ROS Image Pipeline](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline) or similar packages.
<!-- Split blockquote -->
    > **Note**: The model name must be `model.onnx`.

5. Rebuild and source `isaac_ros_dope`:

   ```bash
   cd /workspaces/isaac_ros-dev
   colcon build --packages-up-to isaac_ros_dope && source install/setup.bash
   ```

6. Start `isaac_ros_dope` using the launch file:

    ```bash
    ros2 launch isaac_ros_dope isaac_ros_dope_triton.launch.py model_name:=Ketchup model_repository_paths:=['/tmp/models'] input_binding_names:=['INPUT__0']   output_binding_names:=['OUTPUT__0'] object_name:=Ketchup
    ```

    > **Note**: `object_name` should correspond to one of the objects listed in the DOPE configuration file, and the specified model should be a DOPE model that is trained for that specific object.

7. Open **another** terminal, and enter the Docker container again:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```

     Then, play the ROS bag:

    ```bash
    ros2 bag play -l src/isaac_ros_pose_estimation/resources/rosbags/dope_rosbag/
    ```

8. Open another terminal window and attach to the same container. You should be able to get the poses of the objects in the images through `ros2 topic echo`:

    In a **third** terminal, enter the Docker container again:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```

    ```bash
    ros2 topic echo /poses
    ```

    > **Note**: We are echoing `/poses` because we remapped the original topic `/dope/pose_array` to `poses` in the launch file.

    Now visualize the pose array in rviz2:

    ```bash
    rviz2
    ```

    Then click on the `Add` button, select `By topic` and choose `PoseArray` under `/poses`. Finally, change the display to show an axes by updating `Shape` to be `Axes`, as shown in the screenshot at the top of this page. Make sure to update the `Fixed Frame` to `camera`.

    > **Note**: For best results, crop/resize input images to the same dimensions your DNN model is expecting.
