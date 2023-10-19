# Isaac ROS Pose Estimation

Deep learned, hardware-accelerated 3D object pose estimation

<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_pose_estimation/dope_objects.png/"><img alt="image" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_pose_estimation/dope_objects.png/" width="400px"/></a></div>

## Overview

[Isaac ROS Pose Estimation](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation)
contains ROS 2 packages to predict the pose of an object.
`isaac_ros_dope` estimates the object’s pose using the 3D
bounding cuboid dimensions of a known object in an input image.
`isaac_ros_centerpose` estimates the object’s pose using the 3D
bounding cuboid dimensions of unknown object instances in a known
category of objects from an input image. `isaac_ros_dope` and
`isaac_ros_centerpose` use GPU acceleration for DNN inference to
estimate the pose of an object. The output prediction can be used by
perception functions when fusing with the corresponding depth to provide
the 3D pose of an object and distance for navigation or manipulation.

<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_pose_estimation_nodegraph.png/"><img alt="image" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_pose_estimation_nodegraph.png/" width="500px"/></a></div>

`isaac_ros_dope` is used in a graph of nodes to estimate the pose of a
known object with 3D bounding cuboid dimensions. To produce the
estimate, a [DOPE](https://github.com/NVlabs/Deep_Object_Pose) (Deep
Object Pose Estimation) pre-trained model is required. Input images may
need to be cropped and resized to maintain the aspect ratio and match
the input resolution of DOPE. After DNN inference has produced an estimate, the
DNN decoder will use the specified object type, along with the belief
maps produced by model inference, to output object poses.

NVLabs has provided a DOPE pre-trained model using the
[HOPE](https://github.com/swtyree/hope-dataset) dataset. HOPE stands
for `Household Objects for Pose Estimation`. HOPE is a research-oriented
dataset that uses toy grocery objects and 3D textured meshes of the objects
for training on synthetic data. To use DOPE for other objects that are
relevant to your application, the model needs to be trained with another
dataset targeting these objects. For example, DOPE has been trained to
detect dollies for use with a mobile robot that navigates under, lifts,
and moves that type of dolly. To train your own DOPE model, please refer to the
[Training your Own DOPE Model Tutorial](https://nvidia-isaac-ros.github.io/concepts/pose_estimation/dope/tutorial_custom_model.html).

`isaac_ros_centerpose` has similarities to `isaac_ros_dope` in that
both estimate an object pose; however, `isaac_ros_centerpose` provides
additional functionality. The
[CenterPose](https://github.com/NVlabs/CenterPose) DNN performs
object detection on the image, generates 2D keypoints for the object,
estimates the 6-DoF pose up to a scale, and regresses relative 3D bounding cuboid
dimensions. This is performed on a known object class without knowing
the instance-for example, a CenterPose model can detect a chair without having trained on
images of that specific chair.

Pose estimation is a compute-intensive task and therefore not performed at the
frame rate of an input camera. To make efficient use of resources,
object pose is estimated for a single frame and used as an input to
navigation. Additional object pose estimates are computed to further
refine navigation in progress at a lower frequency than the input rate
of a typical camera.

Packages in this repository rely on accelerated DNN model inference
using [Triton](https://github.com/triton-inference-server/server) or
[TensorRT](https://developer.nvidia.com/tensorrt) from [Isaac ROS DNN Inference](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference).
For preprocessing, packages in this rely on the `Isaac ROS DNN Image Encoder`,
which can also be found at [Isaac ROS DNN Inference](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/blob/main/isaac_ros_dnn_image_encoder).

## Performance

| Sample Graph<br/><br/>                                                                                                                                          | Input Size<br/><br/>    | AGX Orin<br/><br/>                                                                                                                                            | Orin NX<br/><br/>                                                                                                                                            | Orin Nano 8GB<br/><br/>                                                                                                                                       | x86_64 w/ RTX 4060 Ti<br/><br/>                                                                                                                                |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [DOPE Pose Estimation Graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/scripts/isaac_ros_dope_graph.py)<br/><br/><br/><br/>             | VGA<br/><br/><br/><br/> | [39.8 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_dope_graph-agx_orin.json)<br/><br/><br/>33 ms<br/><br/>        | [17.3 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_dope_graph-orin_nx.json)<br/><br/><br/>120 ms<br/><br/>       | –<br/><br/><br/><br/>                                                                                                                                         | [89.2 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_dope_graph-nuc_4060ti.json)<br/><br/><br/>15 ms<br/><br/>       |
| [Centerpose Pose Estimation Graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/scripts/isaac_ros_centerpose_graph.py)<br/><br/><br/><br/> | VGA<br/><br/><br/><br/> | [36.1 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_centerpose_graph-agx_orin.json)<br/><br/><br/>5.7 ms<br/><br/> | [19.4 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_centerpose_graph-orin_nx.json)<br/><br/><br/>7.4 ms<br/><br/> | [13.8 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_centerpose_graph-orin_nano.json)<br/><br/><br/>12 ms<br/><br/> | [50.2 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_centerpose_graph-nuc_4060ti.json)<br/><br/><br/>14 ms<br/><br/> |

---

## Documentation

Please visit the [Isaac ROS Documentation](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/index.html) to learn how to use this repository.

---

## Packages

* [`isaac_ros_centerpose`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_centerpose/index.html)
  * [Quickstart](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_centerpose/index.html#quickstart)
  * [Troubleshooting](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_centerpose/index.html#troubleshooting)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_centerpose/index.html#api)
* [`isaac_ros_dope`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_dope/index.html)
  * [Quickstart](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_dope/index.html#quickstart)
  * [Try More Examples](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_dope/index.html#try-more-examples)
  * [Use Different Models](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_dope/index.html#use-different-models)
  * [Troubleshooting](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_dope/index.html#troubleshooting)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_dope/index.html#api)

## Latest

Update 2023-10-18: Adding NITROS CenterPose decoder.
