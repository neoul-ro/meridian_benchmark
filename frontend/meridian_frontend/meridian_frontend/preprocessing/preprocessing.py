#!/usr/bin/env python3
"""RGB+aligned depth 토픽을 구독해 상주 GPU 버퍼(rgb_raw·depth_m)에 올리는 노드.
하류(sam/clip)는 구독 없이 이 버퍼를 주소로 직접 읽음 — 단일 프로세스 조립 전제(Tegra CUDA IPC 미지원).
계약: rgb_raw (480,640,3) uint8 항상 RGB · depth_m (480,640) float32 meters · 하류 전달은 sink 직접 호출."""
import os
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
import message_filters
import numpy as np
import torch


class Preprocessing:
    def __init__(self, node, color_topic=None, depth_topic=None):
        self.node = node
        self.log = node.get_logger().get_child('pre')
        # 조립 공용 stream, 최고 우선순위(값 작을수록 높음)
        _lo, _hi = torch.cuda.Stream.priority_range()
        torch.cuda.set_stream(torch.cuda.Stream(priority=_hi))
        node.declare_parameter('color_topic', color_topic or '/camera/color/image_raw')
        node.declare_parameter('depth_topic', depth_topic or '/camera/aligned_depth_to_color/image_raw')
        # 상주 GPU 버퍼, 하류가 주소로 직접 읽음
        self.rgb_raw = torch.zeros(480, 640, 3, dtype=torch.uint8, device='cuda')   # 항상 RGB
        self.depth_m = torch.zeros(480, 640, dtype=torch.float32, device='cuda')    # meters
        # pinned staging, 주소 고정(graph memcpy 소스) · depth 는 int16 비트뷰
        self.c_pin = torch.empty(480, 640, 3, dtype=torch.uint8, pin_memory=True)
        self.d_pin = torch.empty(480, 640, dtype=torch.int16, pin_memory=True)
        self.c_np = self.c_pin.numpy()                        # pinned 공유 뷰, np.copyto 대상
        self.d_np = self.d_pin.numpy()
        self.c_stage = torch.empty(480, 640, 3, dtype=torch.uint8, device='cuda')
        self.d_stage = torch.empty(480, 640, dtype=torch.int16, device='cuda')
        self.graph = None                                     # lazy capture, 첫 콜백에서(init 캡처 시 SIGSEGV)
        self.ev0 = torch.cuda.Event(enable_timing=True)       # GPU 계측 + pinned 재사용 보호
        self.ev1 = torch.cuda.Event(enable_timing=True)
        self.have_ev = False
        self.sink = None                                      # 조립 시 sam.step 연결
        self.seq = 0
        self.stamp = None
        self.checked = False
        self.swap = False                                     # bgr8 입력이면 True, capture 전 확정
        self.t_cpu = 0.0
        self.t_gpu = 0.0
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        sub_c = message_filters.Subscriber(node, Image, node.get_parameter('color_topic').value, qos_profile=qos)
        sub_d = message_filters.Subscriber(node, Image, node.get_parameter('depth_topic').value, qos_profile=qos)
        self.sync = message_filters.ApproximateTimeSynchronizer([sub_c, sub_d], queue_size=2, slop=0.03)
        self.sync.registerCallback(self.on_pair)
        self.log.info('up — pinned+graph(lazy) 업로드, GPU 상주 rgb(480,640,3)u8 · depth(480,640)f32')

    def capture(self):
        for _ in range(3):                                    # 웜업(allocator 안정화)
            self.c_stage.copy_(self.c_pin, non_blocking=True)
            self.rgb_raw.copy_(self.c_stage.flip(-1) if self.swap else self.c_stage)
            self.d_stage.copy_(self.d_pin, non_blocking=True)
            self.depth_m.copy_((self.d_stage.to(torch.int32) & 0xFFFF).to(torch.float32).mul_(0.001))
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.c_stage.copy_(self.c_pin, non_blocking=True)
            self.rgb_raw.copy_(self.c_stage.flip(-1) if self.swap else self.c_stage)
            self.d_stage.copy_(self.d_pin, non_blocking=True)
            self.depth_m.copy_((self.d_stage.to(torch.int32) & 0xFFFF).to(torch.float32).mul_(0.001))  # int16 뷰→uint16 복원, mm→m
        torch.cuda.synchronize()
        self.log.info(f'upload graph captured (lazy, 첫 프레임) — 채널 flip={self.swap}')

    def on_pair(self, cmsg, dmsg):
        if not self.checked:
            if cmsg.height != 480 or cmsg.width != 640 or dmsg.height != 480 or dmsg.width != 640:
                self.log.fatal(f'해상도 불일치: color {cmsg.width}x{cmsg.height} depth {dmsg.width}x{dmsg.height} (요구 640x480)')
                return
            if cmsg.encoding not in ('rgb8', 'bgr8'):
                self.log.fatal(f'미지원 color encoding: {cmsg.encoding} (rgb8/bgr8만)')
                return
            if dmsg.encoding not in ('16UC1', 'mono16'):
                self.log.fatal(f'미지원 depth encoding: {dmsg.encoding} (16UC1 mm 요구)')
                return
            self.swap = (cmsg.encoding == 'bgr8')             # graph 캡처 전 확정(flip 이 graph 에 고정)
            self.log.info(f'입력 계약 확인 — color {cmsg.encoding} (flip={self.swap}) · depth {dmsg.encoding}')
            self.checked = True
        if self.graph is None:
            self.capture()
        t0 = self.node.get_clock().now()
        if self.have_ev:
            self.ev1.synchronize()                            # 직전 replay 완료 후 pinned 재사용
            self.t_gpu += self.ev0.elapsed_time(self.ev1)
        np.copyto(self.c_np, np.frombuffer(cmsg.data, dtype=np.uint8).reshape(480, 640, 3))
        np.copyto(self.d_np, np.frombuffer(dmsg.data, dtype=np.uint16).reshape(480, 640).view(np.int16))
        self.ev0.record()
        self.graph.replay()                                   # sync 없음, stream 순서로 보장
        self.ev1.record()
        self.have_ev = True
        self.seq += 1
        self.stamp = cmsg.header.stamp
        self.t_cpu += (self.node.get_clock().now() - t0).nanoseconds / 1e6
        if self.seq % 300 == 0:
            self.log.info(f'seq={self.seq} upload cpu avg {self.t_cpu/300:.3f} ms | gpu(graph) avg {self.t_gpu/300:.3f} ms')
            self.t_cpu = 0.0; self.t_gpu = 0.0
        if self.sink is not None:
            self.sink(self.seq, self.stamp)                   # 직접 호출, seq·stamp 명시 전달

