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
#
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess


# Model and engine configuration.
MESH_FILE_NAME = 'textured_simple.obj'

REFINE_MODEL_NAME = 'dummy_refine_model.onnx'
REFINE_ENGINE_NAME = 'dummy_refine_trt_engine.plan'
SCORE_MODEL_NAME = 'dummy_score_model.onnx'
SCORE_ENGINE_NAME = 'dummy_score_trt_engine.plan'

REFINE_ENGINE_PATH = '/tmp/' + REFINE_ENGINE_NAME
SCORE_ENGINE_PATH = '/tmp/' + SCORE_ENGINE_NAME


def generate_tensorrt_engine(engine_path, model_name, trtexec_args):
    """
    Generate TensorRT engine file from ONNX model using trtexec.

    Parameters
    ----------
    engine_path : str
        Path where the engine file will be saved
    model_name : str
        Name of the model for logging purposes
    trtexec_args : list
        List of trtexec command arguments

    """
    if not os.path.isfile(engine_path):
        print(f'Generating an engine file for the {model_name} model...')
        # Prepend 'trtexec' to the arguments list.
        cmd = ['/usr/src/tensorrt/bin/trtexec'] + trtexec_args
        print('Generating model engine file by command: ', ' '.join(cmd))
        result = subprocess.run(
            cmd,
            env=os.environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            raise Exception(
                f'Failed to convert with status: {result.returncode}.\n'
                f'stderr:\n' + result.stderr.decode('utf-8')
            )
        print(f'{model_name} model engine file generation was finished')


def generate_foundationpose_engines():
    """
    Generate both Refine and Score TensorRT engine files for FoundationPose tests.

    This function should be called at the beginning of generate_test_description()
    in FoundationPose test files.
    """
    # Get correct model paths (relative to test directory).
    base_path = os.path.dirname(__file__)
    refine_model_path = os.path.join(base_path, '../../test/models', REFINE_MODEL_NAME)
    score_model_path = os.path.join(base_path, '../../test/models', SCORE_MODEL_NAME)

    # Generate Refine engine.
    refine_trtexec_args = [
        f'--onnx={refine_model_path}',
        f'--saveEngine={REFINE_ENGINE_PATH}',
        '--minShapes=input1:1x160x160x6,input2:1x160x160x6',
        '--optShapes=input1:1x160x160x6,input2:1x160x160x6',
        '--maxShapes=input1:42x160x160x6,input2:42x160x160x6',
        '--fp16',
        '--skipInference',
    ]
    generate_tensorrt_engine(REFINE_ENGINE_PATH, 'Refine', refine_trtexec_args)

    # Generate Score engine.
    score_trtexec_args = [
        f'--onnx={score_model_path}',
        f'--saveEngine={SCORE_ENGINE_PATH}',
        '--fp16',
        '--minShapes=input1:1x160x160x6,input2:1x160x160x6',
        '--optShapes=input1:1x160x160x6,input2:1x160x160x6',
        '--maxShapes=input1:252x160x160x6,input2:252x160x160x6',
        '--skipInference',
    ]
    generate_tensorrt_engine(SCORE_ENGINE_PATH, 'Score', score_trtexec_args)


def get_engines():
    """
    Get model and engine names and paths for FoundationPose tests.

    Returns
    -------
    dict
        Dictionary containing model and engine names and paths

    """
    # Calculate correct model paths (relative to test directory).
    base_path = os.path.dirname(__file__)
    refine_model_path = os.path.join(base_path, '../../test/models', REFINE_MODEL_NAME)
    score_model_path = os.path.join(base_path, '../../test/models', SCORE_MODEL_NAME)

    return {
        'refine_model_name': REFINE_MODEL_NAME,
        'refine_engine_name': REFINE_ENGINE_NAME,
        'score_model_name': SCORE_MODEL_NAME,
        'score_engine_name': SCORE_ENGINE_NAME,
        'refine_model_path': refine_model_path,
        'refine_engine_path': REFINE_ENGINE_PATH,
        'score_model_path': score_model_path,
        'score_engine_path': SCORE_ENGINE_PATH,
    }
