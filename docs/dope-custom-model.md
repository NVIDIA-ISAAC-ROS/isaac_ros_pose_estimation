# Training your own DOPE model

## Overview

The DOPE network architecture is intended to be trained on objects of a specific class, which means that using DOPE for pose estimation of a custom object class requires training a custom model for that class.

[NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) offers a convenient workflow for training a custom DOPE model using synthetic data generation (SDG).

## Tutorial Walkthrough

1. Clone the [Isaac Sim DOPE Training repository](https://github.com/andrewyguo/dope_training#deep-object-pose-estimation-dope---training) and follow the training instructions to prepare a custom DOPE model.
2. Using the [Isaac Sim DOPE inference script](https://github.com/andrewyguo/dope_training/tree/master/inference), test the custom DOPE model's inference capability and ensure that the quality is acceptable for your use case.

3. Follow steps 1-5 of the main DOPE [quickstart](../README.md#quickstart).

4. At step 6, move the prepared `.pth` model output from the Isaac Sim DOPE Training script into the `/tmp/models` path inside the Docker container.
    ```bash
    docker cp custom_model.pth isaac_ros_dev-x86_64-container:/tmp/models
    ```
5. At step 7, run the `dope_converter.py` script with the custom model:

    ```bash
    python3 /workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation/isaac_ros_dope/scripts/dope_converter.py --format onnx --input /tmp/models/custom_model.pth
    ```

6. Proceed through steps 8-9.
7. At step 10, launch the ROS 2 launchfile with the custom model:

    ```bash
    ros2 launch isaac_ros_dope isaac_ros_dope_tensor_rt.launch.py model_file_path:=/tmp/models/custom_model.onnx engine_file_path:=/tmp/models/custom_model.plan
    ```

8. Continue with the rest of the quickstart. You should now be able to detect poses of custom objects.
