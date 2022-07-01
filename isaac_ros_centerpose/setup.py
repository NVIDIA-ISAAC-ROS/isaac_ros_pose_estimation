# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from glob import glob
import os

from setuptools import setup

package_name = 'isaac_ros_centerpose'

setup(
    name=package_name,
    version='0.9.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Jeff Smith',
    author_email='jeffreys@nvidia.com',
    maintainer='Ethan Yu',
    maintainer_email='ethany@nvidia.com',
    description='CenterPose: Pose Estimation using Deep Learning',
    license='NVIDIA Isaac ROS Software License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'CenterPoseDecoder = isaac_ros_centerpose.CenterPoseDecoder:main'
        ],
    },
)
