#!/usr/bin/env python3
# pose.py — 카메라 world pose 공급 노드.
# /pose (PoseStamped) = 카메라 optical frame 의 T_world_cam, header.stamp 는 이미지와 같은 시계.
# pose_at(stamp) = stamp 이하 최신 샘플 1개, 보간·외삽 없음.
import numpy as np
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy

CAP = 2048                                                # 링버퍼 (200Hz × 10s)


def quat_to_mat(q):
    """q = (x, y, z, w) → (3,3) 회전행렬."""
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w),     s * (x * z + y * w)],
        [s * (x * y + z * w),     1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w),     s * (y * z + x * w),     1 - s * (x * x + y * y)],
    ])


class Pose:
    def __init__(self, node, pose_topic=None):
        self.node = node
        self.log = node.get_logger().get_child('pose')
        node.declare_parameter('pose_topic', pose_topic or '/pose')
        self.buf_t = np.zeros(CAP, dtype=np.int64)         # header stamp [ns]
        self.buf_p = np.zeros((CAP, 3))
        self.buf_q = np.zeros((CAP, 4))                    # (x, y, z, w)
        self.n = 0                                         # 총 수신 수
        self.frame = None
        qos = QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT)
        node.create_subscription(PoseStamped, node.get_parameter('pose_topic').value, self.on_pose, qos)
        self.log.info(f'up — pose={node.get_parameter("pose_topic").value} (PoseStamped = T_world_cam)')

    def on_pose(self, m):
        i = self.n % CAP
        self.buf_t[i] = m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec
        p, q = m.pose.position, m.pose.orientation
        self.buf_p[i] = (p.x, p.y, p.z)
        self.buf_q[i] = (q.x, q.y, q.z, q.w)
        if self.frame is None:
            self.frame = m.header.frame_id
            self.log.info(f'pose 수신 시작 — frame_id={m.header.frame_id!r}')
        self.n += 1

    def pose_at(self, stamp):
        """stamp 이하 최신 샘플의 T_world_cam = (R(3,3), t(3,)). 보간 없음, 미준비면 None."""
        if self.n < 1:
            return None
        m = min(self.n, CAP)
        ts = self.buf_t[:m]
        want = stamp.sec * 1_000_000_000 + stamp.nanosec
        past = ts <= want
        if not past.any():
            return None                                    # 이미지보다 과거인 pose 가 아직 없음
        j = int(np.where(past, ts, np.iinfo(np.int64).min).argmax())
        return quat_to_mat(self.buf_q[j]), self.buf_p[j].copy()

    def gap_ms(self, stamp):
        """stamp 에 가장 가까운 pose 샘플과의 시간차 [ms]."""
        if self.n == 0:
            return float('nan')
        m = min(self.n, CAP)
        ts = self.buf_t[:m]
        want = stamp.sec * 1_000_000_000 + stamp.nanosec
        return float(np.min(np.abs(ts - want)) / 1e6)

