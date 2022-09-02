# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from enum import IntEnum

import cv2
import numpy as np
import torch
from torch import nn


def tensor_to_numpy_array(tensor):
    shape = tuple(tensor.shape.dims)
    x = None
    if tensor.data_type == 9:  # float32
        x = np.frombuffer(bytearray(tensor.data), dtype='float32')
    elif tensor.data_type == 10:  # float64
        x = np.frombuffer(bytearray(tensor.data), dtype='float64')
    else:
        print('Received tensor of incorrect type:', tensor.data_type)
        return None
    x = np.reshape(x, shape)
    return x


def quaternion_from_axis_rotation(axis, theta):
    # Normalize axis vector if needed
    if not np.isclose(np.linalg.norm(axis), 1.0):
        axis = axis / np.linalg.norm(axis)

    half_theta = theta * 0.5
    sht = np.sin(half_theta)

    # Quaternion order is (x, y, z, w)
    qq = np.array([sht*axis[0], sht*axis[1], sht*axis[2], np.cos(half_theta)])
    return qq.flatten()


def quaternion_cross(q1, q2):
    q1x, q1y, q1z, q1w = q1
    q2x, q2y, q2z, q2w = q2

    return np.array([q1x * q2w + q1y * q2z - q1z * q2y + q1w * q2x,
                     -q1x * q2z + q1y * q2w + q1z * q2x + q1w * q2y,
                     q1x * q2y - q1y * q2x + q1z * q2w + q1w * q2z,
                     -q1x * q2x - q1y * q2y - q1z * q2z + q1w * q2w],
                    dtype=q1.dtype)


def gather_feat(feat, ind, mask=None):
    dim = feat.shape[2]
    ind = np.broadcast_to(np.expand_dims(
        ind, 2), (ind.shape[0], ind.shape[1], dim))
    # make ind writable
    ind.setflags(write=1)
    feat = torch.gather(torch.from_numpy(feat), 1,
                        torch.from_numpy(ind)).numpy()

    if mask is not None:
        mask = np.broadcast_to(np.expand_dims(mask, 2), (feat.shape))
        feat = feat[mask]
        feat = feat.view(-1, dim)

    return feat


def np_topk(a, k, axis=-1, is_sorted=True):
    if axis is None:
        axis_size = a.size
    else:
        axis_size = a.shape[axis]
    assert 1 <= k <= axis_size

    a = np.asanyarray(a)
    index_array = np.argpartition(a, axis_size-k, axis=axis)
    topk_indices = np.take(index_array, -np.arange(k)-1, axis=axis)
    topk_values = np.take_along_axis(a, topk_indices, axis=axis)
    if is_sorted:
        sorted_indices_in_topk = np.argsort(topk_values, axis=axis)
        sorted_indices_in_topk = np.flip(sorted_indices_in_topk, axis=axis)
        sorted_topk_values = np.take_along_axis(
            topk_values, sorted_indices_in_topk, axis=axis)
        sorted_topk_indices = np.take_along_axis(
            topk_indices, sorted_indices_in_topk, axis=axis)
        return sorted_topk_values, sorted_topk_indices
    return topk_values, topk_indices


def topk(scores, K=40):
    batch, cat, height, width = scores.shape
    topk_scores, topk_inds = np_topk(np.reshape(scores, (batch, cat, -1)), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).astype(float)
    topk_xs = (topk_inds % width).astype(float)

    topk_score, topk_ind = np_topk(np.reshape(topk_scores, (batch, -1)), K)
    topk_clses = (topk_ind // K).astype(int)
    topk_inds = np.reshape(gather_feat(np.reshape(
        topk_inds, (batch, -1, 1)), topk_ind), (batch, K))
    topk_ys = np.reshape(gather_feat(np.reshape(
        topk_ys, (batch, -1, 1)), topk_ind), (batch, K))
    topk_xs = np.reshape(gather_feat(np.reshape(
        topk_xs, (batch, -1, 1)), topk_ind), (batch, K))

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def topk_channel(scores, K=40):
    batch, cat, height, width = scores.shape

    topk_scores, topk_inds = np_topk(np.reshape(scores, (batch, cat, -1)), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).astype(float)
    topk_xs = (topk_inds % width).astype(float)

    return topk_scores, topk_inds, topk_ys, topk_xs


def nms(heat, kernel=3):
    m = nn.MaxPool2d(kernel, stride=1, padding=1)
    hmax = m(torch.from_numpy(heat))
    keep = 1.0 * (hmax.numpy() == heat)
    return heat * keep


def transpose_and_gather_feat(feat, ind):
    feat = np.transpose(feat, (0, 2, 3, 1))
    feat = np.reshape(feat, (feat.shape[0], -1, feat.shape[3]))
    feat = gather_feat(feat, ind)
    return feat


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def object_pose_post_process(dets, c, s, h, w):
    # Scale bbox & pts
    ret = []
    for i in range(dets.shape[0]):
        bbox = transform_preds(
            dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
        pts = transform_preds(
            dets[i, :, 5:21].reshape(-1, 2), c[i], s[i], (w, h))

        top_preds = np.concatenate(
            [bbox.reshape(-1, 4), dets[i, :, 4:5],
             pts.reshape(-1, 16),
             dets[i, :, 21:25]
             ], axis=1).astype(np.float32).tolist()
        ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
    return ret


def merge_outputs(detections):
    results = {}
    results[1] = np.concatenate(
        [detection[1] for detection in detections], axis=0).astype(np.float32)
    results[1] = results[1].tolist()
    return results


class CuboidVertexType(IntEnum):
    """Define an object's local coordinate system."""

    FrontTopRight = 0
    FrontTopLeft = 1
    FrontBottomLeft = 2
    FrontBottomRight = 3
    RearTopRight = 4
    RearTopLeft = 5
    RearBottomLeft = 6
    RearBottomRight = 7
    Center = 8
    TotalCornerVertexCount = 8  # Corner vertices don't include the center point
    TotalVertexCount = 9


# List of the vertex indexes in each edge of the cuboid
CuboidLineIndexes = [
    # Front face
    [CuboidVertexType.FrontTopLeft,      CuboidVertexType.FrontTopRight],
    [CuboidVertexType.FrontTopRight,     CuboidVertexType.FrontBottomRight],
    [CuboidVertexType.FrontBottomRight,  CuboidVertexType.FrontBottomLeft],
    [CuboidVertexType.FrontBottomLeft,   CuboidVertexType.FrontTopLeft],
    # Back face
    [CuboidVertexType.RearTopLeft,       CuboidVertexType.RearTopRight],
    [CuboidVertexType.RearTopRight,      CuboidVertexType.RearBottomRight],
    [CuboidVertexType.RearBottomRight,   CuboidVertexType.RearBottomLeft],
    [CuboidVertexType.RearBottomLeft,    CuboidVertexType.RearTopLeft],
    # Left face
    [CuboidVertexType.FrontBottomLeft,   CuboidVertexType.RearBottomLeft],
    [CuboidVertexType.FrontTopLeft,      CuboidVertexType.RearTopLeft],
    # Right face
    [CuboidVertexType.FrontBottomRight,  CuboidVertexType.RearBottomRight],
    [CuboidVertexType.FrontTopRight,     CuboidVertexType.RearTopRight],
]


class Cuboid3d():
    """This class defines a 3D cuboid."""

    # Create a box with a certain size
    def __init__(self, size3d=[1.0, 1.0, 1.0]):
        self.center_location = [0, 0, 0]
        self.size3d = size3d
        self._vertices = [0, 0, 0] * CuboidVertexType.TotalCornerVertexCount
        self.generate_vertices()

    def get_vertex(self, vertex_type):
        """Return the location of a vertex.

        Args:
            vertex_type: enum of type CuboidVertexType
        Returns:
            Numpy array(3) - Location of the vertex type in the cuboid
        """
        return self._vertices[vertex_type]

    def get_vertices(self):
        return self._vertices

    def generate_vertices(self):
        width, height, depth = self.size3d

        # By default just use the normal OpenCV coordinate system
        cx, cy, cz = self.center_location
        # X axis point to the right
        right = cx + width / 2.0
        left = cx - width / 2.0
        # Y axis point upward
        top = cy + height / 2.0
        bottom = cy - height / 2.0
        # Z axis point forward
        front = cz + depth / 2.0
        rear = cz - depth / 2.0

        # List of 8 vertices of the box
        self._vertices = [
            [left, bottom, rear],    # Rear Bottom Left
            [left, bottom, front],   # Front Bottom Left
            [left, top, rear],       # Rear Top Left
            [left, top, front],      # Front Top Left

            [right, bottom, rear],   # Rear Bottom Right
            [right, bottom, front],  # Front Bottom Right
            [right, top, rear],      # Rear Top Right
            [right, top, front],     # Front Top Right
        ]


class CuboidPNPSolver(object):
    """
    This class is used to find the 6-DoF pose of a cuboid given its projected vertices.

    Runs perspective-n-point (PNP) algorithm.
    """

    # Class variables
    cv2version = cv2.__version__.split('.')
    cv2majorversion = int(cv2version[0])

    def __init__(self,
                 scaling_factor=1,
                 camera_intrinsic_matrix=None,
                 cuboid3d=None,
                 dist_coeffs=np.zeros((4, 1)),
                 min_required_points=4
                 ):
        self.min_required_points = max(4, min_required_points)
        self.scaling_factor = scaling_factor

        if camera_intrinsic_matrix is not None:
            self._camera_intrinsic_matrix = camera_intrinsic_matrix
        else:
            self._camera_intrinsic_matrix = np.array([
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ])
        self._cuboid3d = cuboid3d
        self._dist_coeffs = dist_coeffs

    def set_camera_intrinsic_matrix(self, new_intrinsic_matrix):
        """Set the camera intrinsic matrix."""
        self._camera_intrinsic_matrix = new_intrinsic_matrix

    def set_dist_coeffs(self, dist_coeffs):
        """Set the camera intrinsic matrix."""
        self._dist_coeffs = dist_coeffs

    def __check_pnp_result(self,
                           points,
                           projected_points,
                           fail_if_projected_diff_exceeds,
                           fail_if_projected_value_exceeds):
        """
        Check whether the output of PNP seems reasonable.

        Inputs:
        - cuboid2d_points:  list of XY tuples
        - projected points:  np.ndarray of np.ndarrays
        """
        p1 = points
        p2 = projected_points.tolist()
        assert len(p1) == len(p2)

        # Compute max Euclidean 2D distance b/w points and projected points
        max_euclidean_dist = 0
        for i in range(len(p1)):
            if p1[i] is not None:
                dist = np.linalg.norm(np.array(p1[i]) - np.array(p2[i]))
                if dist > max_euclidean_dist:
                    max_euclidean_dist = dist

        # Compute max projected absolute value
        max_abs_value = 0
        for i in range(len(p2)):
            assert len(p2[i]) == 2
            for val in p2[i]:
                if val > max_abs_value:
                    max_abs_value = val

        # Return success (true) or failure (false)
        return max_euclidean_dist <= fail_if_projected_diff_exceeds \
            and max_abs_value <= fail_if_projected_value_exceeds

    def solve_pnp(self,
                  cuboid2d_points,
                  pnp_algorithm=None,
                  fail_if_projected_diff_exceeds=250,
                  fail_if_projected_value_exceeds=1e5
                  ):
        """
        Detect the rotation and traslation of a cuboid object.

        Inputs:
        - cuboid2d_points:  list of XY tuples
          ...

        Outputs:
        - location in 3D
        - pose in 3D (as quaternion)
        - projected points:  np.ndarray of np.ndarrays

        """
        # Fallback to default PNP algorithm base on OpenCV version
        if pnp_algorithm is None:
            if CuboidPNPSolver.cv2majorversion == 2:
                pnp_algorithm = cv2.CV_ITERATIVE
            elif CuboidPNPSolver.cv2majorversion == 3:
                pnp_algorithm = cv2.SOLVEPNP_ITERATIVE
                # Alternative algorithms:
                # pnp_algorithm = SOLVE_PNP_P3P
                # pnp_algorithm = SOLVE_PNP_EPNP
            else:
                pnp_algorithm = cv2.SOLVEPNP_EPNP

        location = None
        quaternion = None
        projected_points = cuboid2d_points
        cuboid3d_points = np.array(self._cuboid3d.get_vertices())
        obj_2d_points = []
        obj_3d_points = []

        for i in range(CuboidVertexType.TotalCornerVertexCount):
            check_point_2d = cuboid2d_points[i]
            # Ignore invalid points
            if (check_point_2d is None):
                continue
            obj_2d_points.append(check_point_2d)
            obj_3d_points.append(cuboid3d_points[i])

        obj_2d_points = np.array(obj_2d_points, dtype=float)
        obj_3d_points = np.array(obj_3d_points, dtype=float)

        valid_point_count = len(obj_2d_points)

        # Can only do PNP if we have more than 3 valid points
        is_points_valid = valid_point_count >= self.min_required_points

        if is_points_valid:
            ret, rvec, tvec = cv2.solvePnP(
                obj_3d_points,
                obj_2d_points,
                self._camera_intrinsic_matrix,
                self._dist_coeffs,
                flags=pnp_algorithm
            )

            if ret:
                # OpenCV result
                location = [x[0] for x in tvec]
                quaternion = self.convert_rvec_to_quaternion(rvec)

                # Use OpenCV to project 3D points
                projected_points, _ = cv2.projectPoints(cuboid3d_points, rvec, tvec,
                                                        self._camera_intrinsic_matrix,
                                                        self._dist_coeffs)
                projected_points = np.squeeze(projected_points)

                success = self.__check_pnp_result(
                    cuboid2d_points,
                    projected_points,
                    fail_if_projected_diff_exceeds,
                    fail_if_projected_value_exceeds)

                # If the location.Z is negative or object is behind the camera,
                # flip both location and rotation
                x, y, z = location
                if z < 0 or not success:
                    # Get the opposite location
                    location = [-x, -y, -z]

                    # Change the rotation by 180 degree
                    rotate_angle = np.pi
                    rotate_quaternion = quaternion_from_axis_rotation(
                        location, rotate_angle)
                    quaternion = quaternion_cross(
                        rotate_quaternion, quaternion)

        # Quaternion ordering: (x, y, z, w)
        return location, \
            [quaternion[0], quaternion[1], quaternion[2], quaternion[3]], \
            projected_points

    def convert_rvec_to_quaternion(self, rvec):
        """Convert rvec (which is log quaternion) to quaternion."""
        theta = np.sqrt(rvec[0] * rvec[0] + rvec[1] *
                        rvec[1] + rvec[2] * rvec[2])  # in radians
        raxis = [rvec[0] / theta, rvec[1] / theta, rvec[2] / theta]

        # Quaternion order is (x, y, z, w)
        return quaternion_from_axis_rotation(raxis, theta)

    def project_points(self, rvec, tvec):
        """Project points from model onto image using rotation, translation."""
        output_points, tmp = cv2.projectPoints(
            self.__object_vertex_coordinates,
            rvec,
            tvec,
            self.__camera_intrinsic_matrix,
            self.__dist_coeffs)

        output_points = np.squeeze(output_points)
        return output_points
