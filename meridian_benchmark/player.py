"""GT player: publishes benchmark input topics from prebuilt GT files.

Reads the GT dir (frames.csv, gt_poses.csv, gt_seg/, gt_frame_clouds/,
gt_embeddings.npz) plus the dataset dir (rgb/depth PNGs, camera_info yaml)
and publishes the selected topics in real time with the original stamps.
All topics of one frame carry a byte-identical header.stamp — upstream
modules join on exact stamps.

Publishing is deferred until every published topic has at least
`expected_subs` subscribers and the counts have been stable for `settle`
seconds (upstream QoS is RELIABLE KEEP_LAST 10 VOLATILE: anything sent
before a subscriber matches is lost for that subscriber).
"""

import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from .ros_io import (DEFAULT_QOS, make_camera_info, make_cloud_xyz,
                     make_image, stamp_from_ns)

TOPIC_NAMES = {'rgb': '/camera/rgb', 'depth': '/camera/depth',
               'info': '/camera/info', 'seg': '/segment_image',
               'pose': '/pose', 'embedding': '/instance_embedding_set',
               'instance3d': '/instance_3d_set'}


class Player(Node):

    def __init__(self):
        super().__init__('bench_player')
        p = self.declare_parameter
        self.dataset = Path(p('dataset', '').value).expanduser()
        self.gt = Path(p('gt', '').value).expanduser()
        self.topics = list(p('topics', ['rgb']).value)
        self.expected = list(p('expected_subs', [1]).value)
        self.rate = float(p('rate', 1.0).value)
        self.start_frame = int(p('start_frame', 0).value)
        self.end_frame = int(p('end_frame', 0).value)  # 0 = all
        self.pose_source = p('pose_source', 'base').value  # base | cam
        self.pose_type = p('pose_type', 'plain').value     # plain | cov
        self.wait_timeout = float(p('wait_timeout', 60.0).value)
        self.settle = float(p('settle', 2.0).value)
        self.hold = float(p('hold', 3.0).value)
        self.world_frame = p('world_frame', 'map').value
        self.manifest_path = p('manifest', '').value

        if not self.gt.is_dir():
            raise RuntimeError(f'gt dir not found: {self.gt}')
        unknown = [t for t in self.topics if t not in TOPIC_NAMES]
        if unknown:
            raise RuntimeError(f'unknown topics {unknown}; '
                               f'valid: {sorted(TOPIC_NAMES)}')
        if len(self.expected) != len(self.topics):
            self.expected = [1] * len(self.topics)

        self._load_gt()

        self.pubs = {}
        for key in self.topics:
            if key == 'info':
                mtype = CameraInfo
            elif key in ('rgb', 'depth', 'seg'):
                mtype = Image
            elif key == 'pose':
                mtype = (PoseWithCovarianceStamped if self.pose_type == 'cov'
                         else PoseStamped)
            else:
                from meridian_msgs.msg import (Instance3DSet,
                                               InstanceEmbeddingSet)
                mtype = (InstanceEmbeddingSet if key == 'embedding'
                         else Instance3DSet)
            self.pubs[key] = self.create_publisher(
                mtype, TOPIC_NAMES[key], DEFAULT_QOS)

    # ---------------- data ----------------

    def _load_gt(self):
        with open(self.gt / 'frames.csv') as f:
            rows = list(csv.DictReader(f))
        self.frames = rows
        self.stamps = np.array([int(r['stamp_ns']) for r in rows], np.int64)
        n = len(rows)
        self.end = self.end_frame if 0 < self.end_frame <= n else n
        if not (0 <= self.start_frame < self.end):
            raise RuntimeError(
                f'bad frame range [{self.start_frame}, {self.end})')

        if 'pose' in self.topics:
            pose = np.genfromtxt(self.gt / 'gt_poses.csv', delimiter=',',
                                 names=True)
            pre = 'base' if self.pose_source == 'base' else 'cam'
            self.pose_t = np.stack([pose[f'{pre}_t{a}'] for a in 'xyz'], 1)
            self.pose_q = np.stack([pose[f'{pre}_q{a}'] for a in 'xyzw'], 1)

        if 'embedding' in self.topics:
            d = np.load(self.gt / 'gt_embeddings.npz')
            self.emb = d['embeddings'].astype(np.float32)
            self.emb_oids = d['gt_object_ids']
            self.emb_model = str(d['embedding_model_id'])
            self.emb_row = {int(o): i for i, o in enumerate(self.emb_oids)}

        if 'info' in self.topics or 'seg' in self.topics \
                or 'rgb' in self.topics or 'depth' in self.topics:
            if not self.dataset.is_dir():
                raise RuntimeError(f'dataset dir not found: {self.dataset}')
        with open(self.dataset / 'camera_info/left_cam.yaml') as f:
            ci = yaml.safe_load(f)
        self.caminfo = {'frame_id': ci['frame_id'], 'width': ci['image_width'],
                        'height': ci['image_height'], 'K': ci['K'],
                        'D': ci['D']}
        self.cam_frame = ci['frame_id']

    def _frame_msgs(self, i):
        ns = int(self.stamps[i])
        row = self.frames[i]
        out = []
        for key in self.topics:
            if key == 'rgb':
                img = cv2.imread(str(self.dataset / row['rgb']))
                out.append((key, make_image(
                    ns, self.cam_frame,
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 'rgb8')))
            elif key == 'depth':
                raw = cv2.imread(str(self.dataset / row['depth']),
                                 cv2.IMREAD_UNCHANGED)
                out.append((key, make_image(
                    ns, self.cam_frame,
                    raw.astype(np.float32) / 1000.0, '32FC1')))
            elif key == 'info':
                out.append((key, make_camera_info(ns, self.caminfo)))
            elif key == 'seg':
                seg = cv2.imread(str(self.gt / 'gt_seg' / f'{i:06d}.png'),
                                 cv2.IMREAD_UNCHANGED)
                out.append((key, make_image(ns, self.cam_frame, seg,
                                            'mono8')))
            elif key == 'pose':
                out.append((key, self._pose_msg(i, ns)))
            elif key in ('embedding', 'instance3d'):
                out.append((key, self._set_msg(key, i, ns)))
        return out

    def _pose_msg(self, i, ns):
        t, q = self.pose_t[i], self.pose_q[i]
        msg = (PoseWithCovarianceStamped() if self.pose_type == 'cov'
               else PoseStamped())
        msg.header.stamp = stamp_from_ns(ns)
        msg.header.frame_id = self.world_frame
        pose = msg.pose.pose if self.pose_type == 'cov' else msg.pose
        pose.position.x, pose.position.y, pose.position.z = map(float, t)
        (pose.orientation.x, pose.orientation.y,
         pose.orientation.z, pose.orientation.w) = map(float, q)
        return msg

    def _set_msg(self, key, i, ns):
        from meridian_msgs.msg import Instance3DSet, InstanceEmbeddingSet
        d = np.load(self.gt / 'gt_frame_clouds' / f'{i:06d}.npz')
        sids = d['segment_ids']
        if key == 'embedding':
            msg = InstanceEmbeddingSet()
            msg.timestamp = stamp_from_ns(ns)
            msg.embedding_model_id = self.emb_model
            msg.segment_ids = sids.astype(np.uint8).tobytes()
            msg.embedding_dim = self.emb.shape[1]
            rows = [self.emb[self.emb_row[int(o)]]
                    for o in d['gt_object_ids']]
            msg.embeddings = (np.stack(rows).ravel().tolist()
                              if rows else [])
            return msg
        msg = Instance3DSet()
        msg.header.stamp = stamp_from_ns(ns)
        msg.header.frame_id = self.world_frame
        msg.segment_ids = sids.astype(np.uint8).tobytes()
        off, pts = d['offsets'], d['points']
        msg.instance_points = [
            make_cloud_xyz(ns, self.world_frame, pts[off[k]:off[k + 1]])
            for k in range(len(sids))]
        return msg

    # ---------------- run ----------------

    def wait_for_subscribers(self):
        deadline = time.monotonic() + self.wait_timeout
        stable_since = None
        last = None
        while rclpy.ok():
            counts = {k: self.pubs[k].get_subscription_count()
                      for k in self.topics}
            ok = all(counts[k] >= e
                     for k, e in zip(self.topics, self.expected))
            now = time.monotonic()
            if counts != last:
                last, stable_since = counts, now
                self.get_logger().info(f'subscribers: {counts} '
                                       f'(want {dict(zip(self.topics, self.expected))})')
            if ok and now - stable_since >= self.settle:
                return True
            if now > deadline:
                self.get_logger().warning(
                    f'wait_timeout: starting anyway with {counts}')
                return False
            time.sleep(0.2)
        return False

    def play(self):
        self.wait_for_subscribers()
        n = self.end - self.start_frame
        self.get_logger().info(
            f'playing frames [{self.start_frame}, {self.end}) '
            f'({n} frames, rate={self.rate})')
        lag_frames = 0
        t0 = time.monotonic()
        s0 = int(self.stamps[self.start_frame])
        for i in range(self.start_frame, self.end):
            if not rclpy.ok():
                break
            msgs = self._frame_msgs(i)  # load before pacing
            target = t0 + (int(self.stamps[i]) - s0) / 1e9 / self.rate
            now = time.monotonic()
            if now < target:
                time.sleep(target - now)
            elif now - target > 0.04:
                lag_frames += 1
            for key, m in msgs:
                self.pubs[key].publish(m)
            if (i + 1 - self.start_frame) % 200 == 0:
                self.get_logger().info(
                    f'  {i + 1 - self.start_frame}/{n} (lag={lag_frames})')
        wall = time.monotonic() - t0
        self.get_logger().info(
            f'done: {n} frames in {wall:.1f}s (lag_frames={lag_frames}); '
            f'holding {self.hold}s')
        if self.manifest_path:
            manifest = Path(self.manifest_path).expanduser()
            manifest.parent.mkdir(parents=True, exist_ok=True)
            with open(manifest, 'w') as f:
                json.dump({
                    'gt': str(self.gt), 'dataset': str(self.dataset),
                    'topics': {k: TOPIC_NAMES[k] for k in self.topics},
                    'rate': self.rate, 'start_frame': self.start_frame,
                    'end_frame': self.end, 'n_frames': n,
                    'stamp_ns_first': int(self.stamps[self.start_frame]),
                    'stamp_ns_last': int(self.stamps[self.end - 1]),
                    'lag_frames': lag_frames, 'wall_s': wall}, f, indent=1)
        time.sleep(self.hold)


def main():
    rclpy.init()
    node = Player()
    try:
        node.play()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
