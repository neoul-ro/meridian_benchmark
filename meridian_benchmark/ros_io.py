"""ROS message <-> numpy helpers for the benchmark nodes.

Only player/recorder import this; the GT builder and scorer stay ROS-free.
All upstream topics are RELIABLE/KEEP_LAST/10 except
/local_object_graph_snapshot (RELIABLE/KEEP_LAST/1 + TRANSIENT_LOCAL).
"""

import numpy as np
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField

DEFAULT_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
SNAPSHOT_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                          history=HistoryPolicy.KEEP_LAST, depth=1,
                          durability=DurabilityPolicy.TRANSIENT_LOCAL)

ENCODING_DTYPE = {'rgb8': (np.uint8, 3), 'mono8': (np.uint8, 1),
                  '32FC1': (np.float32, 1)}


def stamp_from_ns(ns):
    ns = int(ns)
    return TimeMsg(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def ns_from_stamp(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def make_image(stamp_ns, frame_id, arr, encoding):
    dtype, ch = ENCODING_DTYPE[encoding]
    arr = np.ascontiguousarray(arr, dtype=dtype)
    msg = Image()
    msg.header.stamp = stamp_from_ns(stamp_ns)
    msg.header.frame_id = frame_id
    msg.height, msg.width = arr.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = False
    msg.step = msg.width * ch * arr.itemsize
    msg.data = arr.tobytes()
    return msg


def image_to_array(msg):
    """Image msg -> numpy array (H,W[,C]) for the encodings we handle."""
    dtype, ch = ENCODING_DTYPE[msg.encoding]
    arr = np.frombuffer(bytes(msg.data), dtype=dtype)
    arr = arr.reshape(msg.height, msg.step // (ch * np.dtype(dtype).itemsize),
                      ch)[:, :msg.width]
    return arr[:, :, 0] if ch == 1 else arr


def make_camera_info(stamp_ns, info):
    """info: dict from UH2Sequence.camera_info-style yaml."""
    msg = CameraInfo()
    msg.header.stamp = stamp_from_ns(stamp_ns)
    msg.header.frame_id = info['frame_id']
    msg.width = int(info['width'])
    msg.height = int(info['height'])
    msg.distortion_model = 'plumb_bob'
    msg.d = [float(x) for x in np.asarray(info['D']).ravel()]
    msg.k = [float(x) for x in np.asarray(info['K']).ravel()]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    K = np.asarray(info['K']).ravel()
    msg.p = [K[0], K[1], K[2], 0.0, K[3], K[4], K[5], 0.0,
             K[6], K[7], K[8], 0.0]
    return msg


def make_cloud_xyz(stamp_ns, frame_id, pts):
    """[N,3] float32 -> PointCloud2 (x,y,z float32, point_step 12)."""
    pts = np.ascontiguousarray(pts, dtype=np.float32).reshape(-1, 3)
    msg = PointCloud2()
    msg.header.stamp = stamp_from_ns(stamp_ns)
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = len(pts)
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * len(pts)
    msg.is_dense = True
    msg.data = pts.tobytes()
    return msg


def cloud_to_xyz(msg):
    """PointCloud2 -> [N,3] float32; handles arbitrary point_step with
    float32 x/y/z fields."""
    off = {f.name: f.offset for f in msg.fields}
    n = msg.width * msg.height
    if n == 0:
        return np.zeros((0, 3), np.float32)
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    raw = raw[:n * msg.point_step].reshape(n, msg.point_step)
    return np.stack(
        [raw[:, off[k]:off[k] + 4].copy().view(np.float32)[:, 0]
         for k in ('x', 'y', 'z')], axis=1)


def sanitize_topic(topic):
    return topic.strip('/').replace('/', '_')
