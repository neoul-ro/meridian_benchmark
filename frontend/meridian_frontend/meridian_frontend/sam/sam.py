#!/usr/bin/env python3
"""Segmentor + Geometric Builder 융합 노드. 고정 shape 구간(전처리·TRT·후처리·역투영)은 CUDA graph 1개로 실행.
출력은 GPU 상주 텐서(masks·valid·conf·bbox_p·area·counts·centroid·depth_valid, 주소 고정)이며 하류가 직접 읽음.
pose·intrinsics 주입은 제자리(copy_)만 허용, 재바인딩 금지. points_world·lanes 는 extract_points() 지연 추출."""
import os
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')   # Jetson inductor 병렬 컴파일 hang 회피
import time

import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo

DEV = 'cuda'


class Sam:
    def __init__(self, node, pre, pose=None, engine=None, info_topic=None, z_min=0.1, z_max=5.0):
        self.node = node
        self.log = node.get_logger().get_child('sam')
        self.pre = pre
        self.pose = pose
        node.declare_parameter('seg_engine', engine)                # 경로는 frontend.engine_path 가 확정
        node.declare_parameter('info_topic', info_topic or '/camera/color/camera_info')
        self.PH, self.PW = 192, 256
        self.CONF_TH, self.NMS_IOU, self.AREA_MIN, self.MIOU = 0.4, 0.7, 256, 0.7
        # AREA_MIN 256 proto px = 원본 40x40 px
        self.K1, self.LANES = 1024, 128
        self.GRAY = 114.0 / 255.0
        self.z_min, self.z_max = float(z_min), float(z_max)
        self.Cg = torch.arange(self.PW, device=DEV).float().view(1, 1, self.PW)
        self.Rg = torch.arange(self.PH, device=DEV).float().view(1, self.PH, 1)
        # graph 캡처 전 상주 버퍼 (주소 고정)
        self.depth_p = torch.zeros(self.PH, self.PW, device=DEV)
        self.Ug = torch.arange(self.PW, device=DEV).float().view(1, self.PW).expand(self.PH, self.PW)
        self.Vg = torch.arange(self.PH, device=DEV).float().view(self.PH, 1).expand(self.PH, self.PW)
        self.Kp = torch.tensor([166.277, 166.277, 128.0, 96.0], device=DEV)   # proto 격자 intrinsics (camera_info 로 갱신)
        self.Rw = torch.eye(3, device=DEV)
        self.tw = torch.zeros(3, device=DEV)
        self.Z0 = torch.zeros((), device=DEV)                 # centroid 축약의 else 값 (0×inf NaN 경로 차단)
        self.pose_pin = torch.empty(12, dtype=torch.float32, pin_memory=True)   # pose 주입 staging(고정 주소)
        self.pose_np = self.pose_pin.numpy()
        self._K_latched = False
        lg = trt.Logger(trt.Logger.ERROR)
        eng = trt.Runtime(lg).deserialize_cuda_engine(open(node.get_parameter('seg_engine').value, 'rb').read())
        self.ctx = eng.create_execution_context()
        self.canvas = torch.full((1, 3, 1024, 1024), self.GRAY, device=DEV)
        self.o0 = torch.empty(1, 37, 21504, device=DEV)
        self.o1 = torch.empty(1, 32, 256, 256, device=DEV)
        self.ctx.set_tensor_address('images', self.canvas.data_ptr())
        self.ctx.set_tensor_address('output0', self.o0.data_ptr())
        self.ctx.set_tensor_address('output1', self.o1.data_ptr())
        self.S = torch.cuda.current_stream().cuda_stream
        self.pre_c = torch.compile(self.pre_step, dynamic=False)
        self.post_c = torch.compile(self.post_all, dynamic=False)
        self.log.info('compile+capture 시작 (~20s, 이 동안 프레임은 자연 드랍)')
        t0 = time.time()
        for _ in range(3):                                    # 컴파일·TRT 지연 할당 웜업
            self.pre_c(); self.ctx.execute_async_v3(self.S); self.post_c()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.pre_c()
            self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            self.OUT = self.post_c()
        torch.cuda.synchronize()
        # 하류 공개 텐서 (graph 출력 슬롯, 주소 고정)
        self.masks = self.OUT['masks']; self.valid = self.OUT['valid']
        self.conf = self.OUT['conf']; self.bbox_p = self.OUT['bbox']; self.area = self.OUT['area']
        self.counts = self.OUT['counts']
        self.centroid = self.OUT['centroid']
        self.depth_valid = self.OUT['depth_valid']
        # 지연 추출 슬롯 (가변 shape), extract_points() 가 채움
        self.points_world = None; self.lanes = None; self.det = None
        self._pts_ready = False
        self.stats_pin = torch.empty(3, dtype=torch.int64, pin_memory=True)   # [n, n_conf, n_cand] 비동기 착지
        self.ev_s = torch.cuda.Event()                        # stats D2H 완료 표시
        self.ev_g0 = torch.cuda.Event(enable_timing=True)     # graph GPU 몫 telemetry
        self.ev_g1 = torch.cuda.Event(enable_timing=True)
        self.have_prev = False
        self.n = 0                                            # 직전 확정 n (1프레임 지연 telemetry)
        self.sink = None                                      # 조립 시 tracker.step 연결
        self.n_nopose = 0
        self.t_accum = 0.0; self.g_accum = 0.0; self.x_accum = 0.0; self.count = 0
        self._info_sub = node.create_subscription(
            CameraInfo, node.get_parameter('info_topic').value, self.on_info,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.log.info(
            f'up — capture {time.time()-t0:.0f}s, graph=[letterbox+depth+engine+post(K{self.K1}/NMS0.7x12/'
            f'LANES{self.LANES}/dedup0.7fp16)+역투영], z=[{self.z_min},{self.z_max}], '
            f'pose={"연결" if self.pose is not None else "항등"}')

    def on_info(self, m):
        """첫 camera_info 로 proto intrinsics latch, 제자리 갱신."""
        if self._K_latched:
            return
        s = m.width / self.PW                                 # 원본 → proto 축소 배율 (설계상 2.5)
        self.Kp.copy_(torch.tensor([m.k[0] / s, m.k[4] / s, m.k[2] / s, m.k[5] / s]))
        self._K_latched = True
        if self._info_sub is not None:
            self.node.destroy_subscription(self._info_sub); self._info_sub = None
        self.log.info(f'intrinsics latch — 원본 {m.width}x{m.height} fx={m.k[0]:.2f} cx={m.k[2]:.2f} '
                               f'→ proto {[round(v, 3) for v in self.Kp.tolist()]}')

    def pre_step(self):
        img = self.pre.rgb_raw.permute(2, 0, 1)[None].float().div(255.0)
        self.canvas[:, :, 128:896, :] = F.interpolate(img, (768, 1024), mode='bilinear', align_corners=False)
        self.depth_p.copy_(F.interpolate(self.pre.depth_m[None, None],
                                         (self.PH, self.PW), mode='nearest')[0, 0])

    def post_all(self):
        p0 = self.o0[0]
        cf, i1 = torch.topk(p0[4], self.K1)                   # 1차 topk (고정 shape)
        bx4 = p0[:4].index_select(1, i1)
        coe = p0[5:].index_select(1, i1)
        x1 = bx4[0] - bx4[2] / 2; y1 = bx4[1] - bx4[3] / 2    # cxcywh -> xyxy
        x2 = bx4[0] + bx4[2] / 2; y2 = bx4[1] + bx4[3] / 2
        ab = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        xx1 = torch.max(x1[:, None], x1[None, :]); yy1 = torch.max(y1[:, None], y1[None, :])
        xx2 = torch.min(x2[:, None], x2[None, :]); yy2 = torch.min(y2[:, None], y2[None, :])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (ab[:, None] + ab[None, :] - inter + 1e-9)
        sup = iou.triu(1) > self.NMS_IOU                      # box NMS, 12회 고정 반복 = greedy 고정점
        okc = cf > self.CONF_TH
        keep = okc.clone()
        for _ in range(12):
            keep = okc & ~(sup & keep.unsqueeze(1)).any(0)
        sc2 = cf * keep                                        # LANES 적재
        v2, i2 = torch.topk(sc2, self.LANES)
        valid = v2 > self.CONF_TH
        bbp = torch.stack([x1, y1, x2, y2], 1).index_select(0, i2) / 4.0
        bbp[:, 1] -= 32.0; bbp[:, 3] -= 32.0
        bbp[:, 0].clamp_(0, self.PW); bbp[:, 2].clamp_(0, self.PW)
        bbp[:, 1].clamp_(0, self.PH); bbp[:, 3].clamp_(0, self.PH)
        co = coe.index_select(1, i2).T.contiguous()
        m = (co @ self.o1[0][:, 32:224, :].reshape(32, -1)).view(self.LANES, self.PH, self.PW)   # mask 조립
        m = (m > 0) & (self.Cg >= bbp[:, 0].view(-1, 1, 1)) & (self.Cg < bbp[:, 2].view(-1, 1, 1)) \
                  & (self.Rg >= bbp[:, 1].view(-1, 1, 1)) & (self.Rg < bbp[:, 3].view(-1, 1, 1))  # bbox 밖 소거
        area = m.view(self.LANES, -1).sum(1)                   # area gate
        valid = valid & (area >= self.AREA_MIN)
        fh = (m & valid.view(-1, 1, 1)).view(self.LANES, -1).half()   # mask dedup
        minter = (fh @ fh.t()).float()
        areaf = area.float() * valid.float()
        miou = minter / (areaf[:, None] + areaf[None, :] - minter).clamp(min=1)
        msup = (miou > self.MIOU) & (v2[:, None] > v2[None, :])
        for _ in range(4):
            valid = valid & ~(msup & valid[:, None]).any(0)
        stats = torch.stack([valid.sum(), (p0[4] > self.CONF_TH).sum(), keep.sum()])   # [n, n_conf, n_cand]
        # 역투영 (고정 shape, graph 안)
        Z = self.depth_p
        P = torch.stack([(self.Ug - self.Kp[2]) / self.Kp[0] * Z,
                         (self.Vg - self.Kp[3]) / self.Kp[1] * Z, Z], -1).view(-1, 3)
        Pw = P @ self.Rw.T + self.tw
        mv = m & valid.view(-1, 1, 1) & ((Z > self.z_min) & (Z < self.z_max)).unsqueeze(0)
        mvf = mv.view(self.LANES, -1)
        counts = mvf.sum(1)
        # centroid 합산 = 축별 2D 축약 (결정론적, 3D 브로드캐스트 금지)
        PwT = Pw.t().contiguous()                          # (3,M) 연속, 열 축약 coalescing
        csum = torch.stack([torch.where(mvf, PwT[0].unsqueeze(0), self.Z0).sum(1),
                            torch.where(mvf, PwT[1].unsqueeze(0), self.Z0).sum(1),
                            torch.where(mvf, PwT[2].unsqueeze(0), self.Z0).sum(1)], 1)
        centroid = csum / counts.clamp(min=1).float().unsqueeze(1)
        # 비율이므로 상한 고정 — 융합 나눗셈이 역수 곱으로 낮춰지면 1 ULP 초과(1.0000001)가 나온다
        depth_valid = (counts.float() / area.clamp(min=1).float()).clamp(max=1.0)
        return dict(masks=m, valid=valid, conf=v2, bbox=bbp, stats=stats,
                    Pw=Pw, mvf=mvf, counts=counts, area=area,
                    centroid=centroid, depth_valid=depth_valid)

    def extract_points(self):
        """점군 지연 추출. 발화 프레임에서 호출, graph 출력은 다음 replay 전까지 유효, 프레임당 1회."""
        if self._pts_ready:
            return
        idx = self.OUT['mvf'].nonzero()
        self.lanes = idx[:, 0]
        self.points_world = self.OUT['Pw'][idx[:, 1]]
        self._pts_ready = True

    def step(self, seq, stamp):
        t0 = self.node.get_clock().now()
        if self.pose is not None:                             # 제자리 주입 (graph 캡처 주소 유지)
            pr = self.pose.pose_at(stamp)
            if pr is None:
                self.n_nopose += 1
                if self.n_nopose <= 3:
                    self.log.warn(f'seq={seq} pose 미준비 — 프레임 건너뜀')
                return
            R, t = pr
            self.pose_np[:9] = np.asarray(R, dtype=np.float64).reshape(-1)
            self.pose_np[9:] = t
            self.Rw.copy_(self.pose_pin[:9].view(3, 3), non_blocking=True)   # 같은 stream 선행 enqueue
            self.tw.copy_(self.pose_pin[9:], non_blocking=True)
        if self.have_prev:                                    # 직전 프레임 telemetry 수확
            self.ev_s.synchronize()
            n, n_conf, n_cand = self.stats_pin.tolist()
            self.n = n
            self.g_accum += self.ev_g0.elapsed_time(self.ev_g1)
            if n_conf >= self.K1 or n_cand >= self.LANES:
                self.log.warn(f'overflow: n_conf={n_conf} n_cand={n_cand} — K1/LANES 재보정 필요')
        self.ev_g0.record()
        self.graph.replay()                                   # enqueue-only — 동기화 없음
        self.ev_g1.record()
        self.stats_pin.copy_(self.OUT['stats'], non_blocking=True)   # 비동기 D2H (다음 step에서 수확)
        self.ev_s.record()
        self.have_prev = True
        self.t_accum += (self.node.get_clock().now() - t0).nanoseconds / 1e6
        # 추출 꼬리: 프레임 유일 동기점 (graph 완료 대기)
        tx = time.perf_counter()
        self.det = self.valid.nonzero().squeeze(1)            # 유효 lane 인덱스, 하류 공유
        self._pts_ready = False
        self.x_accum += (time.perf_counter() - tx) * 1e3
        self.count += 1
        if self.count % 100 == 0:
            self.log.info(
                f'seq={seq} n={self.n} det={int(self.det.numel())} | enqueue cpu {self.t_accum/100:.3f} ms | '
                f'graph gpu {self.g_accum/100:.2f} ms | 추출(동기 포함) {self.x_accum/100:.2f} ms')
            self.t_accum = 0.0; self.g_accum = 0.0; self.x_accum = 0.0
        if self.sink is not None:
            self.sink(seq, stamp)                             # 동일 프레임 보장 직접 호출
