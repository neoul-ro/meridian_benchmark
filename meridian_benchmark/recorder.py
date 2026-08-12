"""Benchmark recorder: subscribes input+output topics, records arrival
times and (for selected topics) payloads to a run directory.

Flowtime is measured externally per the plan: the scorer subtracts the
recorder's receive time of a module's input from that of its output at the
same stamp, so recv_mono_ns is captured first thing in every callback.

Outputs under --out:
  arrivals.csv                    topic, stamp_ns, recv_mono_ns, recv_wall_ns, seq
  pose.csv                        rows for pose kinds
  <topic>/<stamp_ns>.png|.npz     payloads for topics listed in `payload`

record entries are "topic=kind"; kinds:
  image pose pose_cov camera_info embedding_set instance3d_set
  tracklet_set decision_set update_set snapshot event
(snapshot subscribes TRANSIENT_LOCAL depth=1 to match graphcore.)
"""

import csv
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from .ros_io import (DEFAULT_QOS, SNAPSHOT_QOS, cloud_to_xyz,
                     image_to_array, ns_from_stamp, sanitize_topic)


def _msg_types():
    from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
    from sensor_msgs.msg import CameraInfo, Image
    types = {'image': Image, 'pose': PoseStamped,
             'pose_cov': PoseWithCovarianceStamped, 'camera_info': CameraInfo}
    try:
        from meridian_msgs.msg import (AssociationDecisionSet,
                                       GraphUpdateEvent, Instance3DSet,
                                       InstanceEmbeddingSet,
                                       LocalObjectGraphSnapshot,
                                       ObjectUpdateSet, TrackletSet)
        types.update({'embedding_set': InstanceEmbeddingSet,
                      'instance3d_set': Instance3DSet,
                      'tracklet_set': TrackletSet,
                      'decision_set': AssociationDecisionSet,
                      'update_set': ObjectUpdateSet,
                      'snapshot': LocalObjectGraphSnapshot,
                      'event': GraphUpdateEvent})
    except ImportError:
        pass
    return types


def _stamp_ns(kind, msg):
    if kind == 'embedding_set':
        return ns_from_stamp(msg.timestamp)
    if kind == 'snapshot':
        return ns_from_stamp(msg.updated_at)
    return ns_from_stamp(msg.header.stamp)


class Recorder(Node):

    def __init__(self):
        super().__init__('bench_recorder')
        self.out = Path(self.declare_parameter('out', '').value).expanduser()
        record = list(self.declare_parameter('record', ['']).value)
        payload = set(self.declare_parameter('payload', ['']).value)
        record = [r for r in record if r]
        payload = {p for p in payload if p}

        if (self.out / 'arrivals.csv').exists():
            raise RuntimeError(
                f'stale run dir: {self.out} already holds a recording. '
                f'Scoring would mix runs - pick a fresh out dir '
                f'(the launch default is timestamped).')
        self.out.mkdir(parents=True, exist_ok=True)
        self._arr_f = open(self.out / 'arrivals.csv', 'w', newline='')
        self._arr = csv.writer(self._arr_f)
        self._arr.writerow(['topic', 'stamp_ns', 'recv_mono_ns',
                            'recv_wall_ns', 'seq'])
        self._pose_f = None
        self._seq = {}

        types = _msg_types()
        self.subs = []
        for entry in record:
            topic, _, kind = entry.partition('=')
            if kind not in types:
                raise RuntimeError(f'unknown/unavailable kind "{kind}" '
                                   f'for {topic} (have: {sorted(types)})')
            save = topic in payload
            if save:
                (self.out / sanitize_topic(topic)).mkdir(exist_ok=True)
            qos = SNAPSHOT_QOS if kind == 'snapshot' else DEFAULT_QOS
            self.subs.append(self.create_subscription(
                types[kind], topic,
                self._make_cb(topic, kind, save), qos))
            self.get_logger().info(
                f'recording {topic} [{kind}]{" +payload" if save else ""}')

    def _make_cb(self, topic, kind, save):
        def cb(msg):
            recv_mono = time.monotonic_ns()
            recv_wall = time.time_ns()
            ns = _stamp_ns(kind, msg)
            seq = self._seq[topic] = self._seq.get(topic, -1) + 1
            self._arr.writerow([topic, ns, recv_mono, recv_wall, seq])
            self._arr_f.flush()
            if kind in ('pose', 'pose_cov'):
                self._write_pose(kind, ns, recv_mono, msg)
            elif save:
                self._write_payload(topic, kind, ns, msg)
        return cb

    def _write_pose(self, kind, ns, recv_mono, msg):
        if self._pose_f is None:
            self._pose_f = open(self.out / 'pose.csv', 'w', newline='')
            self._pose = csv.writer(self._pose_f)
            self._pose.writerow(['stamp_ns', 'recv_mono_ns', 'tx', 'ty', 'tz',
                                 'qx', 'qy', 'qz', 'qw'])
        p = msg.pose.pose if kind == 'pose_cov' else msg.pose
        self._pose.writerow(
            [ns, recv_mono, p.position.x, p.position.y, p.position.z,
             p.orientation.x, p.orientation.y, p.orientation.z,
             p.orientation.w])
        self._pose_f.flush()

    def _write_payload(self, topic, kind, ns, msg):
        d = self.out / sanitize_topic(topic)
        if kind == 'image':
            arr = image_to_array(msg)
            if msg.encoding == 'rgb8':
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            if msg.encoding == '32FC1':
                np.save(d / f'{ns}.npy', arr)
            else:
                cv2.imwrite(str(d / f'{ns}.png'), arr)
        elif kind == 'instance3d_set':
            sids = np.frombuffer(bytes(msg.segment_ids), np.uint8)
            clouds = [cloud_to_xyz(pc) for pc in msg.instance_points]
            offsets = np.cumsum([0] + [len(c) for c in clouds])
            np.savez_compressed(
                d / f'{ns}.npz', segment_ids=sids,
                offsets=offsets.astype(np.int64),
                points=(np.concatenate(clouds) if clouds
                        else np.zeros((0, 3), np.float32)))
        elif kind == 'embedding_set':
            sids = np.frombuffer(bytes(msg.segment_ids), np.uint8)
            emb = np.asarray(msg.embeddings, np.float32)
            emb = emb.reshape(len(sids), msg.embedding_dim) \
                if len(sids) and msg.embedding_dim else emb.reshape(0, 0)
            np.savez_compressed(d / f'{ns}.npz', segment_ids=sids,
                                embeddings=emb,
                                model_id=np.str_(msg.embedding_model_id))
        # other kinds: arrivals only (TBD backend scoring)

    def close(self):
        self._arr_f.close()
        if self._pose_f:
            self._pose_f.close()


def main():
    rclpy.init()
    node = Recorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
