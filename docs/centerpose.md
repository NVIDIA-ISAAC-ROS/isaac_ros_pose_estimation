### Inference on CenterPose using Triton
This tutorial is for using CenterPose with Triton.
> **Warning**: These steps will only work on `x86_64` and **NOT** on `Jetson`.

1. Complete steps 1-5 of the quickstart [here](../README.md#quickstart)
2. Select a CenterPose model by visiting the CenterPose model collection available on the official [CenterPose GitHub](https://github.com/NVlabs/CenterPose) repository [here](https://drive.google.com/drive/folders/1QIxcfKepOR4aktOz62p3Qag0Fhm0LVa0). The model is assumed to be downloaded to `~/Downloads` outside the docker container. This example will use `shoe_resnet_140.pth`, which should be downloaded into `/tmp/models` inside the docker container:
    > **Note**: this should be run outside the container
    ```bash
    cd ~/Downloads && \
    docker cp shoe_resnet_140.pth isaac_ros_dev-x86_64-container:/tmp/models
    ```

    > **Warning**: The models in the root directory of the model collection listed above will *NOT WORK* with our inference nodes because they have custom layers not supported by TensorRT nor Triton. Make sure to use the PyTorch weights that have the string `resnet` in their file names.

3. Create a models repository with version `1`:
   ```bash
   mkdir -p /tmp/models/centerpose_shoe/1
   ```

4. Create a configuration file for this model at path `/tmp/models/centerpose_shoe/config.pbtxt`. Note that name has to be the same as the model repository name. Take a look at the example at `isaac_ros_centerpose/test/models/centerpose_shoe/config.pbtxt` and copy that file to `/tmp/models/centerpose_shoe/config.pbtxt`.
   ```bash
   cp /workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation/isaac_ros_centerpose/test/models/centerpose_shoe/config.pbtxt /tmp/models/centerpose_shoe/config.pbtxt
   ```

5.  To run the TensorRT engine plan, convert the PyTorch model to ONNX first. Export the model into an ONNX file using the script provided under `/workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation/isaac_ros_centerpose/scripts/centerpose_pytorch2onnx.py`:
      ```bash
      python3 /workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation/isaac_ros_centerpose/scripts/centerpose_pytorch2onnx.py --input /tmp/models/shoe_resnet_140.pth --output /tmp/models/centerpose_shoe/1/model.onnx
      ```
6.  To get a TensorRT engine plan file with Triton, export the ONNX model into an TensorRT engine plan file using the builtin TensorRT converter `trtexec`:
      ```bash
      /usr/src/tensorrt/bin/trtexec --onnx=/tmp/models/centerpose_shoe/1/model.onnx --saveEngine=/tmp/models/centerpose_shoe/1/model.plan
      ```

7.  Inside the container, build and source the workspace:
    ```bash
    cd /workspaces/isaac_ros-dev && \
      colcon build --symlink-install && \
      source install/setup.bash
    ```

8.  Start `isaac_ros_centerpose` using the launch file:
    ```bash
    ros2 launch isaac_ros_centerpose isaac_ros_centerpose.launch.py model_name:=centerpose_shoe model_repository_paths:=['/tmp/models']
    ```

    Then open **another** terminal, and enter the Docker container again:
    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```
     Then, play the ROS bag:
    
    ```bash
    ros2 bag play -l src/isaac_ros_pose_estimation/resources/rosbags/centerpose_rosbag/
    ```

9.  Open another terminal window and attach to the same container. You should be able to get the poses of the objects in the images through `ros2 topic echo`:
    
    In a **third** terminal, enter the Docker container again:
    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```
    ```bash
    source install/setup.bash && \
      ros2 topic echo /object_poses
    ```

10. Launch `rviz2`. Click on `Add` button, select "By topic", and choose `MarkerArray` under `/object_poses`. Set the fixed frame to `centerpose`. You'll be able to see the cuboid marker representing the object's pose detected!

<div align="center"><img src="../resources/centerpose_rviz.png" width="600px"/></div>
