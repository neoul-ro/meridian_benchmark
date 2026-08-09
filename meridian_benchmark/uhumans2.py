"""Reader for the uHumans2 unpacked layout (see meridian_dataset/README.md).

No ROS dependency: plain files only. All poses are returned as
(t[3], q[4] xyzw) pairs or 4x4 matrices, world_T_x convention.
"""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

DEPTH_SCALE = 1000.0  # uint16 mm -> metres


def _read_index(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    stamps = np.array([int(r['stamp_ns']) for r in rows], dtype=np.int64)
    files = [r['filename'] for r in rows]
    return stamps, files


def pack_rgb(img_bgr):
    """BGR uint8 image -> uint32 (r<<16 | g<<8 | b) per pixel."""
    b = img_bgr[:, :, 0].astype(np.uint32)
    g = img_bgr[:, :, 1].astype(np.uint32)
    r = img_bgr[:, :, 2].astype(np.uint32)
    return (r << 16) | (g << 8) | b


def unpack_rgb(packed):
    packed = np.asarray(packed, dtype=np.uint32)
    return np.stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255],
                    axis=-1).astype(np.uint8)  # RGB order


class UH2Sequence:

    def __init__(self, root):
        self.root = Path(root)
        with open(self.root / 'metadata.json') as f:
            self.meta = json.load(f)

        rgb_s, self.rgb_files = _read_index(self.root / 'index/left_cam_rgb.csv')
        dep_s, self.depth_files = _read_index(self.root / 'index/depth_cam_mono.csv')
        seg_s, self.seg_files = _read_index(self.root / 'index/seg_cam_rgb.csv')
        if not (np.array_equal(rgb_s, dep_s) and np.array_equal(rgb_s, seg_s)):
            raise ValueError('rgb/depth/seg stamps differ; frame key assumption broken')
        self.stamps_ns = rgb_s
        self.n_frames = len(rgb_s)

        self._odom = None
        self._tf_static = None

    # ---------------- camera ----------------

    def camera_info(self, cam='left_cam'):
        with open(self.root / f'camera_info/{cam}.yaml') as f:
            d = yaml.safe_load(f)
        K = np.array(d['K'], dtype=np.float64).reshape(3, 3)
        return {'K': K, 'width': d['image_width'], 'height': d['image_height'],
                'frame_id': d['frame_id'], 'D': np.array(d['D'], dtype=np.float64)}

    def rgb(self, i):
        """RGB uint8 HxWx3."""
        img = cv2.imread(str(self.root / self.rgb_files[i]))
        if img is None:
            raise IOError(f'failed to read {self.rgb_files[i]}')
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def depth(self, i):
        """float32 metres, 0 = invalid."""
        raw = cv2.imread(str(self.root / self.depth_files[i]), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise IOError(f'failed to read {self.depth_files[i]}')
        return raw.astype(np.float32) / DEPTH_SCALE

    def seg_packed(self, i):
        """uint32 HxW, (r<<16|g<<8|b) instance colour per pixel."""
        img = cv2.imread(str(self.root / self.seg_files[i]))
        if img is None:
            raise IOError(f'failed to read {self.seg_files[i]}')
        return pack_rgb(img)

    # ---------------- poses ----------------

    def odom(self):
        """(stamps_ns[N], t[N,3], q[N,4] xyzw), world -> base_link_gt."""
        if self._odom is None:
            data = np.genfromtxt(self.root / 'odom.csv', delimiter=',', names=True)
            stamps = data['stamp_ns'].astype(np.int64)
            t = np.stack([data['tx'], data['ty'], data['tz']], axis=1)
            q = np.stack([data['qx'], data['qy'], data['qz'], data['qw']], axis=1)
            order = np.argsort(stamps)
            self._odom = (stamps[order], t[order], q[order])
        return self._odom

    def tf_static(self):
        """{(frame_id, child_frame_id): (t[3], q[4] xyzw)} (last wins on duplicates)."""
        if self._tf_static is None:
            out = {}
            with open(self.root / 'tf/tf_static.csv') as f:
                for r in csv.DictReader(f):
                    key = (r['frame_id'], r['child_frame_id'])
                    t = np.array([float(r['tx']), float(r['ty']), float(r['tz'])])
                    q = np.array([float(r['qx']), float(r['qy']),
                                  float(r['qz']), float(r['qw'])])
                    out[key] = (t, q)
            self._tf_static = out
        return self._tf_static

    def camera_poses(self, stamps_ns=None, cam='left_cam'):
        """world_T_cam at the given stamps (default: all frame stamps).

        Odom (200 Hz) is interpolated to the camera stamps: linear for
        translation, slerp for rotation, clamped at the ends. Pass cam=None
        for the base pose itself (world_T_base, no extrinsic applied).
        Returns (t[N,3], q[N,4] xyzw).
        """
        if stamps_ns is None:
            stamps_ns = self.stamps_ns
        stamps_ns = np.asarray(stamps_ns, dtype=np.int64)

        os_, ot, oq = self.odom()
        s = np.clip(stamps_ns, os_[0], os_[-1]).astype(np.float64)

        t_wb = np.stack([np.interp(s, os_.astype(np.float64), ot[:, k])
                         for k in range(3)], axis=1)
        slerp = Slerp(os_.astype(np.float64), Rotation.from_quat(oq))
        r_wb = slerp(s)
        if cam is None:
            return t_wb, r_wb.as_quat()

        t_bc, q_bc = self.tf_static()[('base_link_gt', cam)]
        r_bc = Rotation.from_quat(q_bc)

        r_wc = r_wb * r_bc
        t_wc = t_wb + r_wb.apply(t_bc)
        return t_wc, r_wc.as_quat()


def pose_matrix(t, q):
    """(t[3], q[4] xyzw) -> 4x4 world_T_x."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q).as_matrix()
    T[:3, 3] = t
    return T


def unproject(depth_m, K, stride=1, valid_mask=None):
    """Depth image -> camera-frame points (optical convention: z forward).

    Returns (points[N,3] float32, (v_idx[N], u_idx[N]) full-res pixel coords).
    """
    H, W = depth_m.shape
    vs = np.arange(0, H, stride)
    us = np.arange(0, W, stride)
    uu, vv = np.meshgrid(us, vs)
    z = depth_m[::stride, ::stride]
    ok = z > 0
    if valid_mask is not None:
        ok &= valid_mask[::stride, ::stride]
    z = z[ok]
    u = uu[ok].astype(np.float64)
    v = vv[ok].astype(np.float64)
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    return pts, (vv[ok], uu[ok])
