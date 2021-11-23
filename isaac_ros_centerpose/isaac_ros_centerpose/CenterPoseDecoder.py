# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from isaac_ros_centerpose.CenterPoseDecoderUtils import Cuboid3d, CuboidPNPSolver, \
    merge_outputs, nms, object_pose_post_process, tensor_to_numpy_array, \
    topk, topk_channel, transpose_and_gather_feat
from isaac_ros_nvengine_interfaces.msg import TensorList
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from scipy import special
import torch
from visualization_msgs.msg import Marker, MarkerArray


# Order and shape of input tensors
TENSOR_IDX = {'hm': 0,
              'wh': 1,
              'hps': 2,
              'reg': 3,
              'hm_hp': 4,
              'hp_offset': 5,
              'scale': 6}

TENSOR_SHAPE = {'hm': (1, 1, 128, 128),
                'wh': (1, 2, 128, 128),
                'hps': (1, 16, 128, 128),
                'hm_hp': (1, 8, 128, 128),
                'reg': (1, 2, 128, 128),
                'hp_offset': (1, 2, 128, 128),
                'scale': (1, 3, 128, 128)}

# Constants
K = 100
CONF_THRESH = 0.3


def decode_impl(hm, wh, kps, hm_hp, reg, hp_offset, obj_scale, K):
    hm = special.expit(hm)
    hm_hp = special.expit(hm_hp)

    batch, _, _, _ = hm.shape
    num_joints = kps.shape[1] // 2

    hm = nms(hm)
    scores, inds, clses, ys, xs = topk(hm, K=K)

    kps = transpose_and_gather_feat(kps, inds)
    # joint offset from the centroid loc
    kps = np.reshape(kps, (batch, K, num_joints*2))
    kps[:, :, ::2] += np.broadcast_to(np.reshape(xs,
                                      (batch, K, 1)), (batch, K, num_joints))
    kps[:, :, 1::2] += np.broadcast_to(np.reshape(ys,
                                       (batch, K, 1)), (batch, K, num_joints))

    reg = transpose_and_gather_feat(reg, inds)
    reg = np.reshape(reg, (batch, K, 2))
    xs = np.reshape(xs, (batch, K, 1)) + reg[:, :, 0:1]
    ys = np.reshape(ys, (batch, K, 1)) + reg[:, :, 1:2]

    wh = transpose_and_gather_feat(wh, inds)
    wh = np.reshape(wh, (batch, K, 2))

    clses = np.reshape(clses, (batch, K, 1)).astype(float)
    scores = np.reshape(scores, (batch, K, 1))

    bboxes = np.concatenate([xs - wh[:, :, 0:1] / 2,
                             ys - wh[:, :, 1:2] / 2,
                             xs + wh[:, :, 0:1] / 2,
                             ys + wh[:, :, 1:2] / 2], axis=2)

    hm_hp = nms(hm_hp)
    kps = np.transpose(np.reshape(
        kps, (batch, K, num_joints, 2)), (0, 2, 1, 3))  # b x J x K x 2
    reg_kps = np.broadcast_to(np.expand_dims(
        kps, 3), (batch, num_joints, K, K, 2))  # b x J x K x K x 2

    hm_score, hm_inds, hm_ys, hm_xs = topk_channel(hm_hp, K=K)  # b x J x K
    if hp_offset is not None:
        hp_offset = transpose_and_gather_feat(
            hp_offset, np.reshape(hm_inds, (batch, -1)))
        hp_offset = np.reshape(hp_offset, (batch, num_joints, K, 2))
        hm_xs = hm_xs + hp_offset[:, :, :, 0]
        hm_ys = hm_ys + hp_offset[:, :, :, 1]
    else:
        hm_xs = hm_xs + 0.5
        hm_ys = hm_ys + 0.5

    thresh = 0.1
    mask = (hm_score > thresh).astype(float)
    hm_score = (1 - mask) * -1 + mask * hm_score  # -1 or hm_score
    hm_ys = (1 - mask) * (-10000) + mask * hm_ys  # -10000 or hm_ys
    hm_xs = (1 - mask) * (-10000) + mask * hm_xs

    hm_kps = np.broadcast_to(np.expand_dims(np.stack([hm_xs, hm_ys], axis=-1), 2),
                             (batch, num_joints, K, K, 2))

    dist = (((reg_kps - hm_kps) ** 2).sum(axis=4) ** 0.5)  # b x J x K x K
    min_dist = dist.min(axis=3)  # b x J x K
    min_ind = dist.argmin(axis=3)  # b x J x K
    hm_score = np.expand_dims(torch.gather(torch.from_numpy(
        hm_score), dim=2, index=torch.from_numpy(min_ind)).numpy(), -1)
    min_dist = np.expand_dims(min_dist, -1)
    min_ind = np.broadcast_to(np.reshape(min_ind, (num_joints, K, 1, 1)),
                              (batch, num_joints, K, 1, 2))
    hm_kps = torch.gather(torch.from_numpy(hm_kps), dim=3,
                          index=torch.from_numpy(min_ind)).numpy()
    hm_kps = np.reshape(hm_kps, (batch, num_joints, K, 2))

    left = np.broadcast_to(np.reshape(
        bboxes[:, :, 0], (batch, 1, K, 1)), (batch, num_joints, K, 1))
    top = np.broadcast_to(np.reshape(
        bboxes[:, :, 1], (batch, 1, K, 1)), (batch, num_joints, K, 1))
    right = np.broadcast_to(np.reshape(
        bboxes[:, :, 2], (batch, 1, K, 1)), (batch, num_joints, K, 1))
    bottom = np.broadcast_to(np.reshape(
        bboxes[:, :, 3], (batch, 1, K, 1)), (batch, num_joints, K, 1))
    mask = (hm_kps[:, :, 0:1] < left) + (hm_kps[:, :, 0:1] > right) + \
           (hm_kps[:, :, 1:2] < top) + (hm_kps[:, :, 1:2] > bottom) + \
           (hm_score < thresh) + (min_dist >
                                  (np.maximum(bottom - top, right - left) * 0.3))
    mask = np.broadcast_to((mask > 0).astype(float), (batch, num_joints, K, 2))
    kps = (1 - mask) * hm_kps + mask * kps
    kps = np.reshape(np.transpose(kps, (0, 2, 1, 3)),
                     (batch, K, num_joints * 2))
    obj_scale = transpose_and_gather_feat(obj_scale, inds)
    obj_scale = np.reshape(obj_scale, (batch, K, 3))

    detections = np.concatenate(
        [bboxes, scores, kps, clses, obj_scale], axis=2)
    return detections


def post_process_impl(res, params_config):
    dets = res.reshape(1, -1, res.shape[2])
    dets = object_pose_post_process(
        dets.copy(),
        [np.divide(params_config['original_image_size'], 2)], [
            params_config['original_image_size'][0]],
        params_config['output_field_size'][0], params_config['output_field_size'][1])
    dets[0][1] = np.array(dets[0][1], dtype=np.float32).reshape(-1, 25)
    results = merge_outputs(dets)

    detected_objects = []

    for bbox in results[1]:
        if bbox[4] > CONF_THRESH:
            pnp_solver = CuboidPNPSolver(
                cuboid3d=Cuboid3d(
                    params_config['height'] * np.array(bbox[22:25]))
            )

            pnp_solver.set_camera_intrinsic_matrix(
                params_config['camera_matrix'])
            points = np.array(bbox[5:21]).reshape(-1, 2)
            points = [(x[0], x[1]) for x in points]
            location, quaternion, projected_points = pnp_solver.solve_pnp(
                points)
            detected_objects.append({
                'location': location,
                'quaternion': quaternion,
                'confidence': bbox[4],
                'cuboid_dimensions': params_config['height'] * np.array(bbox[22:25]),
            })

    return detected_objects


def decode(t_hm, t_wh, t_kps, t_hm_hp, t_reg, t_hp_offset, t_obj_scale, K, params_config):
    # Convert tensors to numpy arrays
    hm = tensor_to_numpy_array(t_hm)
    wh = tensor_to_numpy_array(t_wh)
    kps = tensor_to_numpy_array(t_kps)
    hm_hp = tensor_to_numpy_array(t_hm_hp)
    reg = tensor_to_numpy_array(t_reg)
    hp_offset = tensor_to_numpy_array(t_hp_offset)
    obj_scale = tensor_to_numpy_array(t_obj_scale)

    # Decode tensors into a set of detections
    res = decode_impl(hm, wh, kps, hm_hp, reg, hp_offset, obj_scale, K)
    # Post-process the detections into 3D cuboid bounding boxes and poses
    detected_objects = post_process_impl(res, params_config)

    objects = MarkerArray()
    for idx, obj in enumerate(detected_objects):
        mm = Marker()
        mm.id = idx
        mm.type = Marker.CUBE
        mm.action = 0  # add a detection
        mm.pose.position.x = obj['location'][0]
        mm.pose.position.y = obj['location'][1]
        mm.pose.position.z = obj['location'][2]
        mm.pose.orientation.x = obj['quaternion'][0]
        mm.pose.orientation.y = obj['quaternion'][1]
        mm.pose.orientation.z = obj['quaternion'][2]
        mm.pose.orientation.w = obj['quaternion'][3]
        mm.scale.x = obj['cuboid_dimensions'][0]
        mm.scale.y = obj['cuboid_dimensions'][1]
        mm.scale.z = obj['cuboid_dimensions'][2]
        mm.lifetime = Duration(nanoseconds=33333333.3333).to_msg()  # 30 fps
        mm.color.r = float(params_config['marker_color'][0])
        mm.color.g = float(params_config['marker_color'][1])
        mm.color.b = float(params_config['marker_color'][2])
        mm.color.a = float(params_config['marker_color'][3])
        objects.markers.append(mm)

    return objects


class IsaacROSCenterposeDecoderNode(Node):

    def __init__(self, name='centerpose_decoder_node'):
        super().__init__(name)
        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera_matrix', None),
                ('original_image_size', None),
                ('output_field_size', None),
                ('height', None),
                ('frame_id', 'centerpose'),
                ('marker_color', None)
            ]
        )
        # Sanity check parameters
        self.params_config = {}
        param_names = ['camera_matrix', 'original_image_size',
                       'output_field_size', 'height', 'marker_color',
                       'frame_id']
        for param_name in param_names:
            self.params_config[param_name] = self.get_parameter(
                param_name).value

        if (self.params_config['camera_matrix'] is None) or \
           (len(self.params_config['camera_matrix']) != 9):
            self.get_logger().warning('No camera matrix specified. '
                                      'Pose estimates will be inaccurate')
            # These are the intrinsic camera parameters used to train the
            # NVIDIA-provided pretrained models.
            self.params_config['camera_matrix'] = np.matrix(
                [[616.078125, 0, 325.8349304199219],
                 [0, 616.1030883789062, 244.4612274169922],
                 [0, 0, 1]])
        else:
            self.params_config['camera_matrix'] = np.matrix(
                self.params_config['camera_matrix']).reshape(3, 3)

        if (self.params_config['original_image_size'] is None) or \
           (len(self.params_config['original_image_size']) != 2):
            self.get_logger().warning('No original image size specified. '
                                      'Pose estimates will be inaccurate')
            self.params_config['original_image_size'] = np.array([640, 480])

        if (self.params_config['output_field_size'] is None) or \
           (len(self.params_config['output_field_size']) != 2):
            self.get_logger().warning('No output field size specified. Assuming 128x128')
            self.params_config['output_field_size'] = np.array([128, 128])

        if self.params_config['height'] is None:
            self.get_logger().warning('No height specified. Assuming 1.0')
            self.params_config['height'] = 1.0

        if self.params_config['marker_color'] is None or \
           (len(self.params_config['marker_color']) != 4):
            self.get_logger().warning(
                'No marker color of correct size specified. Assuming RGBA of (1.0, 0.0, 0.0, 1.0)')
            self.params_config['marker_color'] = [1.0, 0.0, 0.0, 1.0]

        # Create the subscriber. This subscriber will subscribe to a TensorList message
        self.subscription_ = self.create_subscription(TensorList, 'tensor_sub',
                                                      self.listener_callback, 10)

        # Create the publisher. This publisher will publish a MarkerArray message
        # to a topic. The queue size is 10 messages.
        self.publisher_ = self.create_publisher(
            MarkerArray, 'object_poses', 10)

    def listener_callback(self, msg):
        tensors = msg.tensors
        if len(tensors) != len(TENSOR_IDX):
            self.get_logger().error('Received tensor not of correct length. '
                                    'Expected %s, got %s' % (len(TENSOR_IDX), len(tensors)))
            return
        for key in TENSOR_IDX.keys():
            if tuple(tensors[TENSOR_IDX[key]].shape.dims) != TENSOR_SHAPE[key]:
                self.get_logger().error(
                    'Tensor %s not of correct shape. Expected %s, got %s' %
                    (key, TENSOR_SHAPE[key], tensors[TENSOR_IDX[key]].shape.dims))
                return

        poses = decode(tensors[TENSOR_IDX['hm']],
                       tensors[TENSOR_IDX['wh']],
                       tensors[TENSOR_IDX['hps']],
                       tensors[TENSOR_IDX['hm_hp']],
                       tensors[TENSOR_IDX['reg']],
                       tensors[TENSOR_IDX['hp_offset']],
                       tensors[TENSOR_IDX['scale']],
                       K, self.params_config)

        if poses is None:
            self.get_logger().error('Error decoding input tensors')
            return

        for pose in poses.markers:
            pose.header = msg.header
            pose.header.frame_id = self.params_config['frame_id']

        # Publish the message to the topic
        self.publisher_.publish(poses)


def main(args=None):
    rclpy.init(args=args)
    node = IsaacROSCenterposeDecoderNode('centerpose_decoder_node')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
