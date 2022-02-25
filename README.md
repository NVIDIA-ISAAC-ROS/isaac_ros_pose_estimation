# Isaac ROS Pose Estimation 

<div align="center"><img src="https://github.com/NVlabs/Deep_Object_Pose/raw/master/dope_objects.png" width="300px"/></div>

## Overview
This repository provides NVIDIA GPU-accelerated packages for 3D object pose estimation. Using a deep learned pose estimation model and a monocular camera, the `isaac_ros_dope` and `isaac_ros_centerpose` package can estimate the 6DOF pose of a target object.

Packages in this repository rely on accelerated DNN model inference using [Triton](https://github.com/triton-inference-server/server) or [TensorRT](https://developer.nvidia.com/tensorrt) from [Isaac ROS DNN Inference](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/tree/main/isaac_ros_dnn_inference).


## System Requirements
This Isaac ROS package is designed and tested to be compatible with ROS2 Foxy on Jetson hardware, in addition to on x86 systems with an Nvidia GPU. On x86 systems, packages are only supported when run in the provided Isaac ROS Dev Docker container.

### Jetson
- AGX Xavier or Xavier NX
- JetPack 4.6

### x86_64 (in Isaac ROS Dev Docker Container)
- CUDA 11.1+ supported discrete GPU
- VPI 1.1.11
- Ubuntu 20.04+

**Note:** For best performance on Jetson, ensure that power settings are configured appropriately ([Power Management for Jetson](https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/power_management_jetson_xavier.html#wwpID0EUHA)).

### Docker
You need to use the Isaac ROS development Docker image from [Isaac ROS Common](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common), based on the version 21.08 image from [Deep Learning Frameworks Containers](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

You must first install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to make use of the Docker container development/runtime environment.

Configure `nvidia-container-runtime` as the default runtime for Docker by editing `/etc/docker/daemon.json` to include the following:
```
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
```
and then restarting Docker: `sudo systemctl daemon-reload && sudo systemctl restart docker`

Run the following script in `isaac_ros_common` to build the image and launch the container on x86_64 or Jetson:

`$ scripts/run_dev.sh <optional_path>`

### Dependencies
- [isaac_ros_common](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/tree/main/isaac_ros_common)
- [isaac_ros_nvengine](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/tree/main/isaac_ros_nvengine)
- [isaac_ros_nvengine_interfaces](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/tree/main/isaac_ros_nvengine_interfaces)
- [isaac_ros_tensor_rt](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/tree/main/isaac_ros_tensor_rt)
- [isaac_ros_triton](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/tree/main/isaac_ros_triton)


## Setup
1. Create a ROS2 workspace if one is not already prepared:
   ```
   mkdir -p your_ws/src
   ```
   **Note**: The workspace can have any name; this guide assumes you name it `your_ws`.
   
2. Clone the Isaac ROS Pose Estimation, Isaac ROS DNN Inference, and Isaac ROS Common package repositories to `your_ws/src`. Check that you have [Git LFS](https://git-lfs.github.com/) installed before cloning to pull down all large files:
   ```
   sudo apt-get install git-lfs
   
   cd your_ws/src   
   git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation
   git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference
   git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common
   ```

3. Start the Docker interactive workspace:
   ```
   isaac_ros_common/scripts/run_dev.sh your_ws
   ```
   After this command, you will be inside of the container at `/workspaces/isaac_ros-dev`. Running this command in different terminals will attach to the same container.

   **Note**: The rest of this README assumes that you are inside this container.

4. Build and source the workspace:
   ```
   cd /workspaces/isaac_ros-dev
   colcon build && . install/setup.bash
   ```
   **Note**: We recommend rebuilding the workspace each time when source files are edited. To rebuild, first clean the workspace by running `rm -r build install log`.

5. (Optional) Run tests to verify complete and correct installation:
   ```
   colcon test --executor sequential
   ```


## Package Reference
### `isaac_ros_dope`
#### Overview
The `isaac_ros_dope` package offers functionality for detecting objects of a specific object type in images and estimating these objects' 6 DOF (degrees of freedom) poses using a trained DOPE (Deep Object Pose Estimation) model. This package sets up pre-processing using the `DNN Image Encoder node`, inference on images by leveraging the `TensorRT node` and provides a decoder that converts the DOPE network's output into an array of 6 DOF poses.

The model provided is taken from the official [DOPE Github repository](https://github.com/NVlabs/Deep_Object_Pose) published by NVIDIA Research. To get a model, visit the PyTorch DOPE model collection [here](https://drive.google.com/drive/folders/1DfoA3m_Bm0fW8tOWXGVxi4ETlLEAgmcg), and use the script under `isaac_ros_dope/scripts` to convert the PyTorch model to ONNX, which can be ingested by the TensorRT node (this script can only be executed on an x86 machine). However, the package should also work if you train your own DOPE model that has an input image size of `[480, 640]`. For instructions to train your own DOPE model, check out the README in the official [DOPE Github repository](https://github.com/NVlabs/Deep_Object_Pose).

#### Package Dependencies
- [isaac_ros_dnn_encoders](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/tree/main/isaac_ros_dnn_encoders)
- [isaac_ros_nvengine_interfaces](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/tree/main/isaac_ros_nvengine_interfaces)
- Inference Packages (can pick either one)
  + [isaac_ros_tensor_rt](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/tree/main/isaac_ros_tensor_rt)
  + [isaac_ros_triton](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/tree/main/isaac_ros_triton)

#### Available Components
| Component         | Topics Subscribed                                                                                       | Topics Published                                                                                                           | Parameters                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ----------------- | ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DopeDecoderNode` | `belief_map_array`: The tensor that represents the belief maps, which are outputs from the DOPE network | `dope/pose_array`: An array of poses of the objects detected by the DOPE network and interpreted by the DOPE decoder node. | `queue_size`: The length of the subscription queues, which is `rmw_qos_profile_default.depth` by default <br>  `frame_id`: The frame ID that the DOPE decoder node will write to the header of its output messages <br>  `configuration_file`: The name of the configuration file to parse. Note: The node will look for that file name under `isaac_ros_dope/config`. By default there is a configuration file under that directory named `dope_config.yaml`. <br>  `object_name`: The object class the DOPE network is detecting and the DOPE decoder is interpreting. This name should be listed in the configuration file along with its corresponding cuboid dimensions. |

#### Configuration
You will need to specify an object type in the `DopeDecoderNode` that is listed in the `dope_config.yaml` file, so the DOPE decoder node will pick the right parameters to transform the belief maps from the inference node to object poses. The `dope_config.yaml` file uses the camera intrinsics of Realsense by default - if you are using a different camera, you will need to modify the `camera_matrix` field with the new, scaled (640x480) camera intrinsics.

### `isaac_ros_centerpose`
#### Overview
The `isaac_ros_centerpose` package offers functionality for detecting objects of a specific class in images and estimating these objects' 6 DOF (degrees of freedom) poses using a trained CenterPose model. Just like DOPE, this package sets up pre-processing using the `DNN Image Encoder node`, inference on images by leveraging an inference node (either `TensorRT` or `Triton` node) and provides a decoder that converts the CenterPose network's output into an array of 6 DOF poses.

The model provided is taken from the official [CenterPose Github repository](https://github.com/NVlabs/CenterPose) published by NVIDIA Research. To get a model, visit the PyTorch CenterPose model collection [here](https://drive.google.com/drive/folders/16HbCnUlCaPcTg4opHP_wQNPsWouUlVZe), and use the script under `isaac_ros_centerpose/scripts` to convert the PyTorch model to ONNX, which can be ingested by the TensorRT node. However, the package should also work if you train your own CenterPose model that has an input image size of `[512, 512]`. For instructions to train your own CenterPose model, check out the README in the official [CenterPose Github repository](https://github.com/NVlabs/CenterPose).

#### Package Dependencies
- [isaac_ros_dnn_encoders](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/tree/main/isaac_ros_dnn_encoders)
- [isaac_ros_nvengine_interfaces](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/tree/main/isaac_ros_nvengine_interfaces)
- Inference Packages (can pick either one)
  + [isaac_ros_tensor_rt](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/tree/main/isaac_ros_tensor_rt)
  + [isaac_ros_triton](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/tree/main/isaac_ros_triton)

#### Available Components
| Component         | Topics Subscribed                                                                                       | Topics Published                                                                                                           | Parameters                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ----------------- | ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CenterPoseDecoderNode` | `tensor_sub`: The TensorList that contains the outputs of the CenterPose network | `object_poses`: A `MarkerArray` representing the poses of objects detected by the CenterPose network and interpreted by the CenterPose decoder node. | `camera_matrix`: A row-major array of 9 floats that represent the camera intrinsics matrix `K`. <br>  `original_image_size`: An array of two floats that represent the size of the original image passed into the `image encoder`. The first element needs to be width, and the second element needs to be height. <br>  `output_field_size`: An array of two integers that represent the size of the 2D keypoint decoding from the network output. The value by default is `[128, 128]`. <br>  `height`: This parameter is used to scale the cuboid used for calculating the size of the objects detected. <br>  `frame_id`: The frame ID that the DOPE decoder node will write to the header of its output messages. The default value is set to `centerpose`. <br>  `marker_color`: An array of 4 floats representing RGBA that will be used to define the color that will be used by RViz to visualize the marker. Each value should be between 0.0 and 1.0. The default value is set to `(1.0, 0.0, 0.0, 1.0)`, which is red. |

#### Configuration
The default parameters for the `CenterPoseDecoderNode` is defined in the `decoders_param.yaml` file under `isaac_ros_centerpose/config`. The `dope_config.yaml` file uses the camera intrinsics of Realsense by default - if you are using a different camera, you will need to modify the `camera_matrix` field.

#### Network Outputs
The CenterPose network has 7 different outputs:
| Output Name | Meaning  |
| ------- | --- |
| hm | Object center heatmap|
| wh | 2D bounding box size |
| hps | Keypoint displacements |
| reg | Sub-pixel offset |
| hm_hp | Keypoint heatmaps |
| hp_offset | Sub-pixel offsets for keypoints |
| scale | Relative cuboid dimensions |

For more context and explanation, you can find the corresponding outputs in Figure 2 of the CenterPose [paper](https://arxiv.org/pdf/2109.06161.pdf) and refer to the paper.
## Walkthroughs
### Inference on DOPE using TensorRT
1. Select a DOPE model by visiting the DOPE model collection available on the official [DOPE GitHub](https://github.com/NVlabs/Deep_Object_Pose) repository [here](https://drive.google.com/open?id=1DfoA3m_Bm0fW8tOWXGVxi4ETlLEAgmcg). For example, download `Ketchup.pth` into `/tmp/models`.

2. In order to run PyTorch models with TensorRT, one option is to export the model into an ONNX file using the script provided under `/workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation/isaac_ros_dope/scripts/dope_converter.py`:
   ```
   python3 /workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation/isaac_ros_dope/scripts/dope_converter.py --format onnx --input /tmp/models/Ketchup.pth
   ```
   The output ONNX file will be located at `/tmp/models/Ketchup.onnx`.

   **Note**: The DOPE decoder currently works with the output of a DOPE network that has a fixed input size of 640 x 480, which are the default dimensions set in the script. In order to use input images of other sizes, make sure to crop/resize using ROS2 nodes from [Isaac ROS Image Pipeline](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline) or similar packages.

3. Modify the following values in the launch file `/workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation/isaac_ros_dope/launch/isaac_ros_dope_tensor_rt.launch.py`:
   ```
   'model_file_path': '/tmp/models/Ketchup.onnx'
   'object_name': 'Ketchup'
   ```
   **Note**: Modify parameters `object_name` and `model_file_path` in the launch file if you are using another model.`object_name` should correspond to one of the objects listed in the DOPE configuration file, and the specified model should be a DOPE model that is trained for that specific object.

4. Rebuild and source `isaac_ros_dope`:
   ```
   cd /workspaces/isaac_ros-dev
   colcon build --packages-up-to isaac_ros_dope && . install/setup.bash
   ```

5. Start `isaac_ros_dope` using the launch file:
   ```
   ros2 launch /workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation/isaac_ros_dope/launch/isaac_ros_dope_tensor_rt.launch.py
   ```

6. Setup `image_publisher` package if not already installed.
   ```
   cd /workspaces/isaac_ros-dev/src 
   git clone --single-branch -b ros2 https://github.com/ros-perception/image_pipeline.git
   cd /workspaces/isaac_ros-dev
   colcon build --packages-up-to image_publisher && . install/setup.bash
   ```

7. Start publishing images to topic `/image` using `image_publisher`, the topic that the encoder is subscribed to.
   ```   
   ros2 run image_publisher image_publisher_node /workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation/resources/0002_rgb.jpg --ros-args -r image_raw:=image
   ```

   <div align="center"><img src="resources/0002_rgb.jpg" width="600px"/></div>

8. Open another terminal window. You should be able to get the poses of the objects in the images through `ros2 topic echo`:
   ```
   source /workspaces/isaac_ros-dev/install/setup.bash
   ros2 topic echo /poses
   ```
   We are echoing the topic `/poses` because we remapped the original topic name `/dope/pose_array` to `/poses` in our launch file.

9. Launch `rviz2`. Click on `Add` button, select "By topic", and choose `PoseArray` under `/poses`. Update "Displays" parameters as shown in the following to see the axes of the object displayed.

   <div align="center"><img src="resources/dope_rviz2.png" width="600px"/></div>

**Note:** For best results, crop/resize input images to the same dimensions your DNN model is expecting.

### Inference on DOPE using Triton
1. Select a DOPE model by visiting the DOPE model collection available on the official [DOPE GitHub](https://github.com/NVlabs/Deep_Object_Pose) repository [here](https://drive.google.com/open?id=1DfoA3m_Bm0fW8tOWXGVxi4ETlLEAgmcg). For example, download `Ketchup.pth` into `/tmp/models/Ketchup`.

2. Setup model repository.

   Create a models repository with version `1`:
   ```
   mkdir -p /tmp/models/Ketchup/1
   ```
   Create a configuration file for this model at path `/tmp/models/Ketchup/config.pbtxt`. Note that name has to be the same as the model repository.
   ```
   name: "Ketchup"
   platform: "onnxruntime_onnx"
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
   - To run ONNX models with Triton, export the model into an ONNX file using the script provided under `/workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation/isaac_ros_dope/scripts/dope_converter.py`:
     ```
     python3 /workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation/isaac_ros_dope/scripts/dope_converter.py --format onnx --input /tmp/models/Ketchup/Ketchup.pth --output /tmp/models/Ketchup/1/model.onnx --input_name INPUT__0 --output_name OUTPUT__0
     ```
     **Note**: The DOPE decoder currently works with the output of a DOPE network that has a fixed input size of 640 x 480, which are the default dimensions set in the script. In order to use input images of other sizes, make sure to crop/resize using ROS2 nodes from [Isaac ROS Image Pipeline](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline) or similar packages. The model name has to be `model.onnx`.

   - To run TensorRT engine plan file with Triton, export the ONNX model into an TensorRT engine plan file using the builtin TensorRT converter `trtexec`:
     ```
     /usr/src/tensorrt/bin/trtexec --onnx=/tmp/models/Ketchup/1/model.onnx --saveEngine=/tmp/models/Ketchup/1/model.plan
     ```
     Modify the following value in `/tmp/models/Ketchup/config.pbtxt`:
     ```
     platform: "tensorrt_plan"
     ```
   - To run PyTorch model with Triton (**inferencing PyTorch model is supported for x86_64 platform only**), the model needs to be saved using `torch.jit.save()`. The downloaded DOPE model is saved with `torch.save()`. Export the DOPE model using the script provided under `/workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation/isaac_ros_dope/scripts/dope_converter.py`:
     ```
     python3 /workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation/isaac_ros_dope/scripts/dope_converter.py --format pytorch --input /tmp/models/Ketchup/Ketchup.pth --output /tmp/models/Ketchup/1/model.pt
     ```
     Modify the following value in `/tmp/models/Ketchup/config.pbtxt`:
     ```
     platform: "pytorch_libtorch"
     ```

3. Modify the following values in the launch file `/workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation/isaac_ros_dope/launch/isaac_ros_dope_triton.launch.py`:
   ```
   'model_name': 'Ketchup'
   'model_repository_paths': ['/tmp/models']
   'input_binding_names': ['INPUT__0']
   'output_binding_names': ['OUTPUT__0']
   'object_name': 'Ketchup'
   ```
   **Note**: `object_name` should correspond to one of the objects listed in the DOPE configuration file, and the specified model should be a DOPE model that is trained for that specific object.

4. Rebuild and source `isaac_ros_dope`:
   ```
   cd /workspaces/isaac_ros-dev
   colcon build --packages-up-to isaac_ros_dope && . install/setup.bash
   ```

5. Start `isaac_ros_dope` using the launch file:
   ```
   ros2 launch /workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation/isaac_ros_dope/launch/isaac_ros_dope_triton.launch.py
   ```

6. Setup `image_publisher` package if not already installed.
   ```
   cd /workspaces/isaac_ros-dev/src
   git clone --single-branch -b ros2 https://github.com/ros-perception/image_pipeline.git
   cd /workspaces/isaac_ros-dev
   colcon build --packages-up-to image_publisher && . install/setup.bash
   ```

7. Start publishing images to topic `/image` using `image_publisher`, the topic that the encoder is subscribed to.
   ```
   ros2 run image_publisher image_publisher_node /workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation/resources/0002_rgb.jpg --ros-args -r image_raw:=image
   ```

8. Open another terminal window. You should be able to get the poses of the objects in the images through `ros2 topic echo`:
   ```
   source /workspaces/isaac_ros-dev/install/setup.bash
   ros2 topic echo /poses
   ```
   We are echoing the topic `/poses` because we remapped the original topic name `/dope/pose_array` to `/poses` in our launch file.

9. Launch `rviz2`. Click on `Add` button, select "By topic", and choose `PoseArray` under `/poses`. Update "Displays" parameters to see the axes of the object displayed. 

<div align="center"><img src="resources/dope_rviz2.png" width="600px"/></div>

**Note:** For best results, crop/resize input images to the same dimensions your DNN model is expecting.

### Inference on CenterPose using Triton
1. Select a CenterPose model by visiting the CenterPose model collection available on the official [CenterPose GitHub](https://github.com/NVlabs/CenterPose) repository [here](https://drive.google.com/drive/folders/1QIxcfKepOR4aktOz62p3Qag0Fhm0LVa0). For example, download `shoe_resnet_140.pth` into `/tmp/models/centerpose_shoe`.

**Note:** The models in the root directory of the model collection listed above will *NOT WORK* with our inference nodes because they have custom layers not supported by TensorRT nor Triton. Make sure to use the PyTorch weights that have the string `resnet` in their file names.

2. Setup model repository.

   Create a models repository with version `1`:
   ```
   mkdir -p /tmp/models/centerpose_shoe/1
   ```

3. Create a configuration file for this model at path `/tmp/models/centerpose_shoe/config.pbtxt`. Note that name has to be the same as the model repository name. Take a look at the example at `isaac_ros_centerpose/test/models/centerpose_shoe/config.pbtxt` and copy that file to `/tmp/models/centerpose_shoe/config.pbtxt`.
   ```
   cp /workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation/isaac_ros_centerpose/test/models/centerpose_shoe/config.pbtxt /tmp/models/centerpose_shoe/config.pbtxt
   ```

4. To run the TensorRT engine plan, convert the PyTorch model to ONNX first. Export the model into an ONNX file using the script provided under `/workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation/isaac_ros_centerpose/scripts/centerpose_pytorch2onnx.py`:
   ```
   python3 /workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation/isaac_ros_centerpose/scripts/centerpose_pytorch2onnx.py --input /tmp/models/centerpose_shoe/shoe_resnet_140.pth --output /tmp/models/centerpose_shoe/1/model.onnx
   ```

5. To get a TensorRT engine plan file with Triton, export the ONNX model into an TensorRT engine plan file using the builtin TensorRT converter `trtexec`:
   ```
   /usr/src/tensorrt/bin/trtexec --onnx=/tmp/models/centerpose_shoe/1/model.onnx --saveEngine=/tmp/models/centerpose_shoe/1/model.plan
   ```

6. Modify the `isaac_ros_centerpose` launch file located in `/workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation/isaac_ros_centerpose/launch/isaac_ros_centerpose.launch.py`. You will need to update the following lines as:
   ```
   'model_name': 'centerpose_shoe',
   'model_repository_paths': ['/tmp/models'],
   ```
   Rebuild and source `isaac_ros_centerpose`:
   ```
   cd /workspaces/isaac_ros-dev
   colcon build --packages-up-to isaac_ros_centerpose && . install/setup.bash
   ```
   Start `isaac_ros_centerpose` using the launch file:
   ```
   ros2 launch isaac_ros_centerpose isaac_ros_centerpose.launch.py
   ```

7. Setup `image_publisher` package if not already installed.
   ```
   cd /workspaces/isaac_ros-dev/src
   git clone --single-branch -b ros2 https://github.com/ros-perception/image_pipeline.git
   cd /workspaces/isaac_ros-dev
   colcon build --packages-up-to image_publisher && . install/setup.bash
   ```

8. Start publishing images to topic `/image` using `image_publisher`, the topic that the encoder is subscribed to.
   ```
   ros2 run image_publisher image_publisher_node /workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation/resources/shoe.jpg --ros-args -r image_raw:=image
   ```

9. Open another terminal window and attach to the same container. You should be able to get the poses of the objects in the images through `ros2 topic echo`:
   ```
   source /workspaces/isaac_ros-dev/install/setup.bash
   ros2 topic echo /object_poses
   ```

10. Launch `rviz2`. Click on `Add` button, select "By topic", and choose `MarkerArray` under `/object_poses`. Set the fixed frame to `centerpose`. You'll be able to see the cuboid marker representing the object's pose detected!

<div align="center"><img src="resources/centerpose_rviz.png" width="600px"/></div>


## Troubleshooting
### Nodes crashed on initial launch reporting shared libraries have a file format not recognized
Many dependent shared library binary files are stored in `git-lfs`. These files need to be fetched in order for Isaac ROS nodes to function correctly.

#### Symptoms
```
/usr/bin/ld:/workspaces/isaac_ros-dev/ros_ws/src/isaac_ros_common/isaac_ros_nvengine/gxf/lib/gxf_jetpack46/core/libgxf_core.so: file format not recognized; treating as linker script
/usr/bin/ld:/workspaces/isaac_ros-dev/ros_ws/src/isaac_ros_common/isaac_ros_nvengine/gxf/lib/gxf_jetpack46/core/libgxf_core.so:1: syntax error
collect2: error: ld returned 1 exit status
make[2]: *** [libgxe_node.so] Error 1
make[1]: *** [CMakeFiles/gxe_node.dir/all] Error 2
make: *** [all] Error 2
```
#### Solution
Run `git lfs pull` in each Isaac ROS repository you have checked out, especially `isaac_ros_common`, to ensure all of the large binary files have been downloaded.

# Updates

| Date       | Changes         |
| ---------- | --------------- |
| 2021-10-20 | Initial release |
