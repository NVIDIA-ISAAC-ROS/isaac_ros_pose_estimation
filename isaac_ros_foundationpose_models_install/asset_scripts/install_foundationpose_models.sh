#!/bin/bash
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Download and convert RT-DETR models to TensorRT engines
# Models will be stored in the isaac_ros_assets directory
# Setup paths
if [ -n "$TENSORRT_COMMAND" ]; then
  # If a custom tensorrt is used, ensure it's lib directory is added to the LD_LIBRARY_PATH
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(readlink -f $(dirname ${TENSORRT_COMMAND})/../../../lib/x86_64-linux-gnu/)"
fi
if [ -z "$ISAAC_ROS_WS" ] && [ -n "$ISAAC_ROS_ASSET_MODEL_PATH" ]; then
  ISAAC_ROS_WS="$(readlink -f $(dirname ${ISAAC_ROS_ASSET_MODEL_PATH})/../../..)"
fi
ASSET_NAME="foundationpose"
MODELS_DIR="${ISAAC_ROS_WS}/isaac_ros_assets/models/${ASSET_NAME}"
EULA_URL="https://catalog.ngc.nvidia.com/orgs/nvidia/teams/isaac/models/dnn_stereo_disparity"
ASSET_DIR="${MODELS_DIR}"
ASSET_INSTALL_PATHS="${ASSET_DIR}/refine_model.onnx ${ASSET_DIR}/score_model.onnx"


source "${ISAAC_ROS_ASSET_EULA_SH:-isaac_ros_asset_eula.sh}"

# Create directories if they don't exist
mkdir -p ${MODELS_DIR}


# Download FoundationPose models
REFINE_MODEL_URL="https://api.ngc.nvidia.com/v2/models/nvidia/isaac/foundationpose/versions/1.0.1_onnx/files/refine_model.onnx"
REFINE_MODEL_ONNX="${MODELS_DIR}/refine_model.onnx"
REFINE_ENGINE_PATH="${MODELS_DIR}/refine_trt_engine.plan"

SCORE_MODEL_URL="https://api.ngc.nvidia.com/v2/models/nvidia/isaac/foundationpose/versions/1.0.1_onnx/files/score_model.onnx"
SCORE_MODEL_ONNX="${MODELS_DIR}/score_model.onnx"
SCORE_ENGINE_PATH="${MODELS_DIR}/score_trt_engine.plan"

# Download refine model
wget -nv -O "${REFINE_MODEL_ONNX}" "${REFINE_MODEL_URL}"

# Download score model
wget -nv -O "${SCORE_MODEL_ONNX}" "${SCORE_MODEL_URL}"

# Convert refine model to TensorRT engine
${TENSORRT_COMMAND:-/usr/src/tensorrt/bin/trtexec} \
    --onnx="${REFINE_MODEL_ONNX}" \
    --saveEngine="${REFINE_ENGINE_PATH}" \
    --minShapes=input1:1x160x160x6,input2:1x160x160x6 \
    --optShapes=input1:1x160x160x6,input2:1x160x160x6 \
    --maxShapes=input1:42x160x160x6,input2:42x160x160x6 > /dev/null 2>&1

# Convert score model to TensorRT engine
${TENSORRT_COMMAND:-/usr/src/tensorrt/bin/trtexec} \
    --onnx="${SCORE_MODEL_ONNX}" \
    --saveEngine="${SCORE_ENGINE_PATH}" \
    --minShapes=input1:1x160x160x6,input2:1x160x160x6 \
    --optShapes=input1:1x160x160x6,input2:1x160x160x6 \
    --maxShapes=input1:252x160x160x6,input2:252x160x160x6 > /dev/null 2>&1
