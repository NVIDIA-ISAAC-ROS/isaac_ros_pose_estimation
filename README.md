# Isaac ROS Pose Estimation

<div align="center"><img src="https://github.com/NVlabs/Deep_Object_Pose/raw/master/dope_objects.png" width="400px"/></div>

## Overview

Isaac ROS Pose Estimation contains ROS 2 packages to predict the pose of an object. `isaac_ros_dope` provides a pose estimation method using 3D bounding cuboid dimensions of a known object in an input image. `isaac_ros_centerpose` provides a pose estimation method using 3D bounding cuboid dimensions of unknown object instances in a known category of objects from an input image. `isaac_ros_dope` and `isaac_ros_centerpose` use GPU acceleration for DNN inference to estimate the pose of an object. The output prediction can be used by perception functions when fusing with a corresponding depth to provide the 3D pose of an object and distance for navigation or manipulation.

<div align="center"><img src="resources/isaac_ros_pose_estimation_nodegraph.png" width="500px"/></div>

`isaac_ros_dope` is used in a graph of nodes to estimate the pose of a known object with 3D bounding cuboid dimensions. To produce the estimate, a [DOPE](https://github.com/NVlabs/Deep_Object_Pose) (Deep Object Pose Estimation) pre-trained model is required. Input images may need to be cropped and resized to maintain the aspect ratio and match the input resolution of DOPE. After DOPE has produced an estimate, the DNN decoder will use the specified object type to transform using belief maps to output object poses.

NVLabs has provided a DOPE pre-trained model using the [HOPE](https://github.com/swtyree/hope-dataset) dataset. HOPE stands for household objects for pose estimation and is a research-oriented dataset using toy grocery objects and 3D textured meshes of the objects for training on synthetic data. To use DOPE for other objects that are relevant to your application, it needs to be trained with another dataset targeting these objects. For example, DOPE has been trained to detect dollies for use with a mobile robot that navigates under, lifts, and moves that type of dolly.

`isaac_ros_centerpose` has similarities to `isaac_ros_dope` in that both estimate an object pose; however, `isaac_ros_centerpose` provides additional functionality. The [CenterPose](https://github.com/NVlabs/CenterPose) DNN performs object detection on the image, generates 2D keypoints for the object, estimates the 6-DoF pose, and regresses relative 3D bounding cuboid dimensions. This is performed on a known object class without knowing the instance--for example, detecting a chair without having trained on images of all chairs. NVLabs has provided pre-trained models for the CenterPose model; however, as with the DOPE model, it needs to be trained with another dataset targeting objects that are specific to your application.  

Pose estimation is a compute-intensive task and not performed at the frame rate of an input camera. To make efficient use of resources, object pose is estimated for a single frame and used as an input to navigation. Additional object pose estimates are computed to further refine navigation in progress at a lower frequency than the input rate of a typical camera.

Packages in this repository rely on accelerated DNN model inference using [Triton](https://github.com/triton-inference-server/server) or [TensorRT](https://developer.nvidia.com/tensorrt) from [Isaac ROS DNN Inference](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference).

## Performance

The following table summarizes the per-platform performance statistics of sample graphs that use this package, with links included to the full benchmark output. These benchmark configurations are taken from the [Isaac ROS Benchmark](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark#list-of-isaac-ros-benchmarks) collection, based on the [`ros2_benchmark`](https://github.com/NVIDIA-ISAAC-ROS/ros2_benchmark) framework.

| Sample Graph                                                                                                                                         | Input Size | AGX Orin                                                                                                                                        | Orin NX                                                                                                                                        | Orin Nano 8GB                                                                                                                                         | x86_64 w/ RTX 4060 Ti                                                                                                                             |
| ---------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| [DOPE Pose Estimation Graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/scripts//isaac_ros_dope_graph.py)             | VGA        | [40.5 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_dope_graph-agx_orin.json)<br>31 ms       | [17.6 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_dope_graph-orin_nx.json)<br>120 ms      | --                                                                                                                                                    | [90.9 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_dope_graph-nuc_4060ti.json)<br>14 ms       |
| [Centerpose Pose Estimation Graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/scripts//isaac_ros_centerpose_graph.py) | VGA        | [50.2 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_centerpose_graph-agx_orin.json)<br>37 ms | [23.7 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_centerpose_graph-orin_nx.json)<br>70 ms | [18.4 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_centerpose_graph-orin_nano_emul.json)<br>87 ms | [45.0 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_centerpose_graph-nuc_4060ti.json)<br>21 ms |

## Table of Contents

- [Isaac ROS Pose Estimation](#isaac-ros-pose-estimation)
  - [Overview](#overview)
  - [Performance](#performance)
  - [Table of Contents](#table-of-contents)
  - [Latest Update](#latest-update)
  - [Supported Platforms](#supported-platforms)
    - [Docker](#docker)
  - [Quickstart](#quickstart)
  - [Next Steps](#next-steps)
    - [Try More Examples](#try-more-examples)
    - [Use Different Models](#use-different-models)
    - [Customize your Dev Environment](#customize-your-dev-environment)
  - [Package Reference](#package-reference)
    - [`isaac_ros_dope`](#isaac_ros_dope)
      - [Usage](#usage)
      - [ROS Parameters](#ros-parameters)
      - [Configuration File](#configuration-file)
      - [ROS Topics Subscribed](#ros-topics-subscribed)
      - [ROS Topics Published](#ros-topics-published)
    - [`isaac_ros_centerpose`](#isaac_ros_centerpose)
      - [Usage](#usage-1)
      - [ROS Parameters](#ros-parameters-1)
      - [Configuration File](#configuration-file-1)
      - [ROS Topics Subscribed](#ros-topics-subscribed-1)
      - [ROS Topics Published](#ros-topics-published-1)
      - [CenterPose Network Output](#centerpose-network-output)
  - [Troubleshooting](#troubleshooting)
    - [Isaac ROS Troubleshooting](#isaac-ros-troubleshooting)
    - [Deep Learning Troubleshooting](#deep-learning-troubleshooting)
  - [Updates](#updates)

## Latest Update

Update 2023-05-25: Performance improvements.

## Supported Platforms

This package is designed and tested to be compatible with ROS 2 Humble running on [Jetson](https://developer.nvidia.com/embedded-computing) or an x86_64 system with an NVIDIA GPU.

> **Note**: Versions of ROS 2 earlier than Humble are **not** supported. This package depends on specific ROS 2 implementation features that were only introduced beginning with the Humble release.

| Platform | Hardware                                                                                                                                                                                                 | Software                                                                                                           | Notes                                                                                                                                                                                   |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Jetson   | [Jetson Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) <br> [Jetson Xavier](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-agx-xavier/) | [JetPack 5.1.1](https://developer.nvidia.com/embedded/jetpack)                                                     | For best performance, ensure that [power settings](https://docs.nvidia.com/jetson/archives/r34.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance.html) are configured appropriately. |
| x86_64   | NVIDIA GPU                                                                                                                                                                                               | [Ubuntu 20.04+](https://releases.ubuntu.com/20.04/) <br> [CUDA 11.8+](https://developer.nvidia.com/cuda-downloads) |

### Docker

To simplify development, we strongly recommend leveraging the Isaac ROS Dev Docker images by following [these steps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/dev-env-setup.md). This will streamline your development environment setup with the correct versions of dependencies on both Jetson and x86_64 platforms.

> **Note**: All Isaac ROS Quickstarts, tutorials, and examples have been designed with the Isaac ROS Docker images as a prerequisite.

## Quickstart

> **Warning**: Step 7 must be performed on `x86_64`. The resultant model should be copied over to the `Jetson`. Also note that the process of model preparation differs significantly from the other repositories.

1. Set up your development environment by following the instructions [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/dev-env-setup.md).
2. Clone this repository and its dependencies under `~/workspaces/isaac_ros-dev/src`.

    ```bash
    cd ~/workspaces/isaac_ros-dev/src
    ```

    ```bash
    git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common
    ```

    ```bash
    git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros
    ```

    ```bash
    git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation
    ```

    ```bash
    git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference
    ```

    ```bash
    git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline
    ```

3. Pull down a ROS Bag of sample data:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation && \
      git lfs pull -X "" -I "resources/rosbags/"
    ```

4. Launch the Docker container using the `run_dev.sh` script:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```

5. Make a directory to place models (inside the Docker container):

   ```bash
   mkdir -p /tmp/models/
   ```

6. Select a DOPE model by visiting the DOPE model collection available on the official [DOPE GitHub](https://github.com/NVlabs/Deep_Object_Pose) repository [here](https://drive.google.com/open?id=1DfoA3m_Bm0fW8tOWXGVxi4ETlLEAgmcg). The model is assumed to be downloaded to `~/Downloads` outside the Docker container.

   This example will use `Ketchup.pth`, which should be downloaded into `/tmp/models` inside the Docker container:
    > **Note**: this should be run outside the Docker container

    On `x86_64`:

    ```bash
    cd ~/Downloads && \
    docker cp Ketchup.pth isaac_ros_dev-x86_64-container:/tmp/models
    ```

7. Convert the PyTorch file into an ONNX file:
    > **Warning**: this step must be performed on `x86_64`. The resultant model will be assumed to have been copied to the `Jetson` in the same output location (`/tmp/models/Ketchup.onnx`)

    ```bash
    python3 /workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation/isaac_ros_dope/scripts/dope_converter.py --format onnx --input /tmp/models/Ketchup.pth
    ```

    If you are planning on using Jetson, copy the generated `.onnx` model into the Jetson, and then copy it over into `aarch64` Docker container.

    We will assume that you already performed the transfer of the model onto the Jetson in the directory `~/Downloads`.

    Enter the Docker container in Jetson:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```

    Make a directory called `/tmp/models` in Jetson:

    ```bash
    mkdir -p /tmp/models
    ```

    **Outside** the container, copy the generated `onnx` model:

    ```bash
    cd ~/Downloads && \
    docker cp Ketchup.onnx isaac_ros_dev-aarch64-container:/tmp/models
    ```

8. Inside the container, build and source the workspace:

    ```bash
    cd /workspaces/isaac_ros-dev && \
      colcon build --symlink-install && \
      source install/setup.bash
    ```

9. (Optional) Run tests to verify complete and correct installation:

    ```bash
    colcon test --executor sequential
    ```

10. Run the following launch files to spin up a demo of this package:

    Launch `isaac_ros_dope`:

    ```bash
    ros2 launch isaac_ros_dope isaac_ros_dope_tensor_rt.launch.py model_file_path:=/tmp/models/Ketchup.onnx engine_file_path:=/tmp/models/Ketchup.plan
    ```

    Then open **another** terminal, and enter the Docker container again:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```

     Then, play the ROS bag:

    ```bash
    ros2 bag play -l src/isaac_ros_pose_estimation/resources/rosbags/dope_rosbag/
    ```

11. Open another terminal window and attach to the same container. You should be able to get the poses of the objects in the images through `ros2 topic echo`:

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

    Then click on the `Add` button, select `By topic` and choose `PoseArray` under `/poses`. Finally, change the display to show an axes by updating `Shape` to be `Axes`, as shown in the screenshot below. Make sure to update the `Fixed Frame` to `camera`.

    <div align="center"><img src="resources/dope_rviz2.png" width="600px"/></div>

    > **Note**: For best results, crop or resize input images to the same dimensions your DNN model is expecting.

## Next Steps

### Try More Examples

To continue your exploration, check out the following suggested examples:

- [`DOPE` with `Triton`](docs/dope-triton.md)
- [`Centerpose` with `Triton`](docs/centerpose.md)
- [`DOPE` with non-standard input image sizes](docs/dope-custom-size.md)
- [Train your own `DOPE` model](docs/dope-custom-model.md)

### Use Different Models

Click [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/blob/main/docs/model-preparation.md) for more information about how to use NGC models.

Alternatively, consult the `DOPE` or `CenterPose` model repositories to try other models.

| Model Name                                                                             | Use Case                                                                               |
| -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| [DOPE](https://drive.google.com/open?id=1DfoA3m_Bm0fW8tOWXGVxi4ETlLEAgmcg)             | The DOPE model repository. This should be used if `isaac_ros_dope` is used             |
| [Centerpose](https://drive.google.com/drive/folders/1QIxcfKepOR4aktOz62p3Qag0Fhm0LVa0) | The Centerpose model repository. This should be used if `isaac_ros_centerpose` is used |

### Customize your Dev Environment

To customize your development environment, reference [this guide](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/README.md).

## Package Reference

### `isaac_ros_dope`

#### Usage

```bash
ros2 launch isaac_ros_dope isaac_ros_dope_tensor_rt.launch.py network_image_width:=<network_image_width> network_image_height:=<network_image_height>
model_file_path:=<model_file_path>
engine_file_path:=<engine_file_path> input_tensor_names:=<input_tensor_names> input_binding_names:=<input_binding_names> input_tensor_formats:=<input_tensor_formats> output_tensor_names:=<output_tensor_names> output_binding_names:=<output_binding_names> output_tensor_formats:=<output_tensor_formats>
tensorrt_verbose:=<tensorrt_verbose> object_name:=<object_name>
```

> **Note**: there is also a `config` file that should be modified in `isaac_ros_dope/config/dope_config.yaml`.

#### ROS Parameters

| ROS Parameter        | Type     | Default            | Description                                                                                                                                                                               |
| -------------------- | -------- | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `configuration_file` | `string` | `dope_config.yaml` | The name of the configuration file to parse. Note: The node will look for that file name under isaac_ros_dope/config                                                                      |
| `object_name`        | `string` | `Ketchup`          | The object class the DOPE network is detecting and the DOPE decoder is interpreting. This name should be listed in the configuration file along with its corresponding cuboid dimensions. |

#### Configuration File

The DOPE configuration file, which can be found at `isaac_ros_dope/config/dope_config.yaml` may need to modified. Specifically, you will need to specify an object type in the `DopeDecoderNode` that is listed in the `dope_config.yaml` file, so the DOPE decoder node will pick the right parameters to transform the belief maps from the inference node to object poses. The `dope_config.yaml` file uses the camera intrinsics of Realsense by default - if you are using a different camera, you will need to modify the camera_matrix field with the new, scaled `(640x480)` camera intrinsics.

> **Note**: The `object_name` should correspond to one of the objects listed in the DOPE configuration file, with the corresponding model used.

#### ROS Topics Subscribed

| ROS Topic          | Interface                                                                                                                                                         | Description                                                                          |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `belief_map_array` | [isaac_ros_tensor_list_interfaces/TensorList](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/isaac_ros_tensor_list_interfaces/msg/TensorList.msg) | The tensor that represents the belief maps, which are outputs from the DOPE network. |

#### ROS Topics Published

| ROS Topic         | Interface                                                                                                        | Description                                                                                             |
| ----------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `dope/pose_array` | [geometry_msgs/PoseArray](https://github.com/ros2/common_interfaces/blob/humble/geometry_msgs/msg/PoseArray.msg) | An array of poses of the objects detected by the DOPE network and interpreted by the DOPE decoder node. |

### `isaac_ros_centerpose`

#### Usage

```bash
ros2 launch isaac_ros_centerpose isaac_ros_centerpose.launch.py network_image_width:=<network_image_width> network_image_height:=<network_image_height> encoder_image_mean:=<encoder_image_mean> encoder_image_stddev:=<encoder_image_stddev>
model_name:=<model_name>
model_repository_paths:=<model_repository_paths> max_batch_size:=<max_batch_size> input_tensor_names:=<input_tensor_names> input_binding_names:=<input_binding_names> input_tensor_formats:=<input_tensor_formats> output_tensor_names:=<output_tensor_names> output_binding_names:=<output_binding_names> output_tensor_formats:=<output_tensor_formats>
```

> **Note**: there is also a `config` file that should be modified in `isaac_ros_centerpose/config/decoders_param.yaml`.

#### ROS Parameters

| ROS Parameter         | Type         | Default                                                                                          | Description                                                                                                                                                                         |
| --------------------- | ------------ | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `camera_matrix`       | `float list` | `[616.078125, 0.0, 325.8349304199219, 0.0, 616.1030883789062, 244.4612274169922, 0.0, 0.0, 1.0]` | A row-major array of 9 floats that represent the camera intrinsics matrix `K`.                                                                                                      |
| `original_image_size` | `float list` | `[640, 480]`                                                                                     | An array of two floats that represent the size of the original image passed into the image encoder. The first element needs to be width, and the second element needs to be height. |
| `output_field_size`   | `int list`   | `[128, 128]`                                                                                     | An array of two integers that represent the size of the 2D keypoint decoding from the network output                                                                                |
| `height`              | `float`      | `0.1`                                                                                            | This parameter is used to scale the cuboid used for calculating the size of the objects detected.                                                                                   |
| `frame_id`            | `string`     | `centerpose`                                                                                     | The frame ID that the DOPE decoder node will write to the header of its output messages                                                                                             |
| `marker_color`        | `float list` | `[1.0, 0.0, 0.0, 1.0]` (red)                                                                     | An array of 4 floats representing RGBA that will be used to define the color that will be used by RViz to visualize the marker. Each value should be between 0.0 and 1.0.           |

#### Configuration File

The default parameters for the `CenterPoseDecoderNode` is defined in the `decoders_param.yaml` file under `isaac_ros_centerpose/config`. The `decoders_param.yaml` file uses the camera intrinsics of RealSense by default - if you are using a different camera, you will need to modify the `camera_matrix` field.

#### ROS Topics Subscribed

| ROS Topic    | Interface                                                                                                                                                         | Description                                                         |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `tensor_sub` | [isaac_ros_tensor_list_interfaces/TensorList](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/isaac_ros_tensor_list_interfaces/msg/TensorList.msg) | The TensorList that contains the outputs of the CenterPose network. |

#### ROS Topics Published

| ROS Topic      | Interface                                                                                                                      | Description                                                                                                                          |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| `object_poses` | [visualization_msgs/MarkerArray](https://github.com/ros2/common_interfaces/blob/humble/visualization_msgs/msg/MarkerArray.msg) | A `MarkerArray` representing the poses of objects detected by the CenterPose network and interpreted by the CenterPose decoder node. |

#### CenterPose Network Output

The CenterPose network has 7 different outputs:
| Output Name | Meaning                         |
| ----------- | ------------------------------- |
| `hm`        | Object center heatmap           |
| `wh`        | 2D bounding box size            |
| `hps`       | Keypoint displacements          |
| `reg`       | Sub-pixel offset                |
| `hm_hp`     | Keypoint heatmaps               |
| `hp_offset` | Sub-pixel offsets for keypoints |
| `scale`     | Relative cuboid dimensions      |

For more context and explanation, see the corresponding outputs in Figure 2 of the CenterPose [paper](https://arxiv.org/pdf/2109.06161.pdf) and refer to the paper.

## Troubleshooting

### Isaac ROS Troubleshooting

For solutions to problems with Isaac ROS, please check [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/troubleshooting.md).

### Deep Learning Troubleshooting

For solutions to problems with using DNN models, please check [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/blob/main/docs/troubleshooting.md).

## Updates

| Date       | Changes                                                                                                  |
| ---------- | -------------------------------------------------------------------------------------------------------- |
| 2023-05-25 | Performance improvements                                                                                 |
| 2023-04-05 | Source available GXF extensions                                                                          |
| 2022-06-30 | Update to use NITROS for improved performance and to be compatible with JetPack 5.0.2                    |
| 2022-06-30 | Refactored README, updated launch file & added `nvidia` namespace, dropped Jetson support for CenterPose |
| 2021-10-20 | Initial update                                                                                           |
