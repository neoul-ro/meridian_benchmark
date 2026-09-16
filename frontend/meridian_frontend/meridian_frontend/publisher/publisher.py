#!/usr/bin/env python3
# publisher.py — frontend 의 유일한 출구. keyframe gate 발화 프레임에서만 tracklet 묶음 1건을 내보낸다.
# 계약: 모든 배열은 tracklet_ids 순서 · tracklet i 의 점 = points 앞에서부터 points_num[i]개씩 · world 좌표
#       embedding 은 tracklet 순서로 N행, L2 정규화 단위 벡터, 미인코딩 행 = 0.
# 구조: 콜백은 D2H enqueue 만. 방출 스레드가 엔진 완료를 기다려 payload 를 조립하고 sink(in-process 인계)와
#       ROS 어댑터에 차례로 넘긴다.
#
# sink 계약 (in-process DA 를 붙이는 쪽이 지켜야 하는 것)
#   1. 호출 스레드 = publisher-worker (executor 스레드 아님).
#   2. payload 중 tracklet_ids·points·depth_valid_ratio·seg_conf 는 pinned 버퍼를 그대로 가리키는 view 이고
#      다음 발화가 덮어쓴다. points_num·id_emb·query_emb 는 복사본이라 그 뒤에도 유효하다. 섞여 있으므로
#      보관하려면 view 쪽을 반드시 복사할 것 — 일부만 낡으면 points_num 오프셋이 다른 프레임 점군을 가리킨다.
#   3. view 는 읽기 전용으로 잠가 둔다. 제자리 수정은 예외가 나며, 그래야 뒤따르는 ROS 어댑터가 오염된 값을
#      내보내지 않는다.
#   4. 시간 예산 = 발화 간격(KF_MIN_GAP 5프레임 = 30Hz 에서 약 167ms). 넘기면 콜백 스레드가 막혀 입력이
#      드랍되고, 0.5s 를 넘기면 pinned 재사용 가드가 RuntimeError 를 내 프로세스가 죽는다.
#   5. sink 예외 = frontend 치명. 방출 스레드가 죽고 그 프레임의 ROS publish 도 실행되지 않는다.
#      조용한 계속보다 명시적 실패를 택한 것.
import array
import queue as _queue
import threading
import time

import numpy as np
import torch
from rclpy.qos import QoSProfile, ReliabilityPolicy

from meridian_frontend_msgs.msg import TrackletSet

DEV = 'cuda'
MAXP = 120000                                              # pinned 점 상한 · 초과 시 경고 후 절단


class TrackletSetPayload:
    """발화 1회 분량의 tracklet 묶음(host 배열).

    수명: tracklet_ids · points · depth_valid_ratio · seg_conf = pinned view(다음 발화가 덮음, 읽기 전용).
          points_num · id_emb · query_emb = 복사본. 보관하려면 view 를 복사할 것.
    wire 와의 차이: 여기서는 points 가 (M,3) · embedding 이 (N,512) 2차원이고 points_num 이 int64 이며 seq 가 있다.
                  ROS 메시지는 전부 평탄 배열이고 points_num 은 uint32, seq 는 없다(header.stamp 로 식별).
    """
    __slots__ = ('seq', 'stamp', 'tracklet_ids', 'points_num', 'points',
                 'depth_valid_ratio', 'seg_conf', 'id_emb', 'query_emb', 'n_enc')

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _f32(a):
    out = array.array('f'); out.frombytes(memoryview(np.ascontiguousarray(a, dtype=np.float32)).cast('B')); return out


def _i64(a):
    out = array.array('q'); out.frombytes(memoryview(np.ascontiguousarray(a, dtype=np.int64)).cast('B')); return out


def _u32(a):
    out = array.array('I'); out.frombytes(memoryview(np.ascontiguousarray(a, dtype=np.uint32)).cast('B')); return out


class RosSink:
    """payload → TrackletSet 메시지 → publish. 프로세스 밖으로 나가는 경로."""

    def __init__(self, node, topic):
        self.pub = node.create_publisher(
            TrackletSet, topic, QoSProfile(depth=4, reliability=ReliabilityPolicy.RELIABLE))
        self.topic_name = self.pub.topic_name

    def __call__(self, p):
        m = TrackletSet()
        m.header.stamp = p.stamp
        m.header.frame_id = 'world'
        m.tracklet_ids = _i64(p.tracklet_ids)
        m.points_num = _u32(p.points_num)
        m.points = _f32(p.points)
        m.depth_valid_ratio = _f32(p.depth_valid_ratio)
        m.seg_conf = _f32(p.seg_conf)
        m.id_emb = _f32(p.id_emb)
        m.query_emb = _f32(p.query_emb)
        self.pub.publish(m)                                # rclpy publish 는 타 스레드 호출 가능


class Publisher:
    def __init__(self, node, sam, tracker, clip=None, topic='~/tracklet_set', sink=None):
        self.node = node
        self.log = node.get_logger().get_child('pub')
        self.sam, self.tracker, self.clip = sam, tracker, clip
        L = sam.LANES
        self.pin_pts = torch.empty(MAXP, 3, dtype=torch.float32, pin_memory=True)
        self.pin_cnt = torch.empty(L, dtype=torch.int64, pin_memory=True)
        self.pin_tid = torch.empty(L, dtype=torch.int64, pin_memory=True)
        self.pin_dv = torch.empty(L, dtype=torch.float32, pin_memory=True)
        self.pin_sc = torch.empty(L, dtype=torch.float32, pin_memory=True)
        self.pin_q = torch.empty(L, 512, dtype=torch.float32, pin_memory=True)
        self.pin_e = torch.empty(L, 512, dtype=torch.float32, pin_memory=True)
        self.ev = torch.cuda.Event()                       # 점군·장부 D2H 완료 (main 스트림)
        self.ev2 = torch.cuda.Event()                      # embedding D2H 완료 (S3)
        self.S3 = torch.cuda.Stream()                      # embedding 정규화·D2H 전용 · 방출 스레드에서만 사용
        self.n_msg = 0; self.n_pts = 0; self.n_obs = 0; self.n_emb = 0
        self.bytes_out = 0; self.n_trunc = 0
        self.t_pub = 0.0; self.t_pub_max = 0.0             # 조립 + ROS publish (sink 제외)
        self.t_sink = 0.0; self.t_sink_max = 0.0           # in-process sink 소요 — 예산 회계를 섞지 않는다
        self.ros = RosSink(node, topic) if topic else None
        self.sink = sink                                   # in-process 인계 훅 · 계약은 파일 상단
        self._q = _queue.Queue(maxsize=2)
        self.job_done = threading.Event(); self.job_done.set()
        self.worker_exc = None                             # 방출 스레드 사망 원인 · check_alive 가 판독
        self.worker = threading.Thread(target=self._wloop, name='publisher-worker', daemon=True)
        self.worker.start()
        self.log.info(f'up — DA 경계 {self.consumers} · 발화 프레임에만 방출 · 조립/인계 = 방출 스레드(콜백 sync 0)')
        if self.ros is None and self.sink is None:
            self.log.warn('소비자가 없음 — 전 파이프라인이 돌지만 결과가 아무 데도 나가지 않음')

    @property
    def consumers(self):
        """사람이 읽는 소비자 구성. 토픽 문자열이 필요하면 self.ros.topic_name 을 쓸 것."""
        out = [self.ros.topic_name] if self.ros is not None else []
        if self.sink is not None:
            out.append('in-process sink')
        return ' + '.join(out) if out else '(소비자 없음)'

    def check_alive(self):
        """방출 스레드 비가동 시 명시적 실패."""
        if self.worker_exc is not None or not self.worker.is_alive():
            raise RuntimeError(f'publisher 방출 스레드 비가동 — 방출 신뢰 불가 (원인: {self.worker_exc!r})')

    # --- 콜백: D2H enqueue + 핸드오프 ---
    def step(self, seq, stamp):
        if not self.tracker.is_kf:
            return
        det = self.sam.det
        N = int(det.numel())
        if N == 0:
            return
        self.check_alive()
        if not self.job_done.wait(timeout=0.5):            # pinned 재사용 전 직전 작업 완료 확인
            raise RuntimeError(f'seq={seq} 직전 방출 작업 미완(0.5s) — pinned 재사용 차단')
        self.sam.extract_points()                          # 지연 추출 · 이미 뽑혔으면 no-op
        cnt = self.sam.counts.index_select(0, det)
        M = int(self.sam.lanes.shape[0])
        if M > MAXP:
            self.n_trunc += 1
            self.log.warn(f'seq={seq} 점 {M} > 상한 {MAXP} — 절단 방출 (MAXP 재보정 필요)')
            M = MAXP
        self.pin_pts[:M].copy_(self.sam.points_world[:M], non_blocking=True)
        self.pin_cnt[:N].copy_(cnt, non_blocking=True)
        self.pin_tid[:N].copy_(self.tracker.tid_cur, non_blocking=True)
        self.pin_dv[:N].copy_(self.sam.depth_valid.index_select(0, det), non_blocking=True)
        self.pin_sc[:N].copy_(self.sam.conf.index_select(0, det), non_blocking=True)
        K = 0; pos = None
        if self.clip is not None and self.clip.seq_emb == seq and self.clip.m > 0:
            K = self.clip.m
            src = self.tracker.emb_pos                     # 인코딩된 lane 의 det 순서 내 위치 · 다음 프레임에 갱신되므로 스냅샷
            pos = np.array(src[:K], dtype=np.int64) if src is not None else np.arange(K)
        self.ev.record()
        self.job_done.clear()
        self._q.put_nowait((seq, stamp, N, M, K, pos))    # 큐 포화 시 예외 · 발화 간격 ≥ KF_MIN_GAP 전제

    @staticmethod
    def _norm(x):
        return x / x.norm(dim=1, keepdim=True).clamp(min=1e-12)

    @staticmethod
    def _ro(a):
        """pinned view 를 읽기 전용으로 잠금 — 소비자의 제자리 수정을 즉시 예외로 만든다."""
        a.flags.writeable = False
        return a

    def _build(self, seq, stamp, N, M, K, pos):
        """host 배열로 payload 조립. view 4종은 읽기 전용, 나머지 3종은 복사본(파일 상단 계약)."""
        cnt = self.pin_cnt[:N].numpy().astype(np.int64)
        if int(cnt.sum()) > M:                             # 절단 시 뒤쪽 tracklet 부터 줄여 합 = M
            cnt = np.diff(np.concatenate(([0], np.minimum(np.cumsum(cnt), M))))
        e = np.zeros((N, 512), np.float32); q = np.zeros((N, 512), np.float32)
        if K > 0:
            e[pos] = self.pin_e[:K].numpy(); q[pos] = self.pin_q[:K].numpy()
        return TrackletSetPayload(
            seq=seq, stamp=stamp, n_enc=K,
            tracklet_ids=self._ro(self.pin_tid[:N].numpy()), points_num=cnt,
            points=self._ro(self.pin_pts[:M].numpy()),
            depth_valid_ratio=self._ro(self.pin_dv[:N].numpy()),
            seg_conf=self._ro(self.pin_sc[:N].numpy()),
            id_emb=e, query_emb=q)

    # --- 방출 스레드 ---
    def _wloop(self):
        try:
            while True:
                seq, stamp, N, M, K, pos = self._q.get()
                t0 = time.perf_counter()
                self.ev.synchronize()
                if K > 0:
                    if not self.clip.enq_done.wait(timeout=0.5):
                        raise RuntimeError(f'seq={seq} clip 워커 enqueue 미완결(0.5s) — 오래된 embedding 방출 차단')
                    self.clip.check_alive()
                    with torch.cuda.stream(self.S3):
                        self.S3.wait_event(self.clip.ev_emb)
                        self.pin_q[:K].copy_(self._norm(self.clip.q[:K]), non_blocking=True)
                        self.pin_e[:K].copy_(self._norm(self.clip.e[:K]), non_blocking=True)
                        self.ev2.record()
                    self.ev2.synchronize()
                p = self._build(seq, stamp, N, M, K, pos)
                if self.sink is not None:                  # in-process 소비자 우선 · 반환 시점까지 view 유효
                    ts = time.perf_counter()
                    self.sink(p)
                    dts = (time.perf_counter() - ts) * 1e3
                    self.t_sink += dts; self.t_sink_max = max(self.t_sink_max, dts)
                    t0 += (time.perf_counter() - ts)       # sink 소요는 frontend 몫에서 제외
                if self.ros is not None:
                    self.ros(p)
                    self.bytes_out += M * 12 + N * 20 + N * 4096
                self.n_msg += 1; self.n_pts += M; self.n_obs += N; self.n_emb += K
                dt = (time.perf_counter() - t0) * 1e3
                self.t_pub += dt; self.t_pub_max = max(self.t_pub_max, dt)
                self.job_done.set()
                self._q.task_done()
        except Exception as e:
            self.worker_exc = e
            self.log.error(f'publisher 방출 스레드 사망: {e!r}')
            self.job_done.set()

    def summary(self):
        n = max(self.n_msg, 1)
        out = [f'publisher: 방출 {self.n_msg} · 관측 {self.n_obs} · 점 {self.n_pts} · embedding {self.n_emb}'
               + (f' · ⚠️절단 {self.n_trunc}' if self.n_trunc else ''),
               f'  소비자 {self.consumers} · 방출당 관측 {self.n_obs/n:.1f} · 점 {self.n_pts/n:.0f} · '
               f'페이로드 {self.bytes_out/n/1024:.0f} KB · 조립+publish {self.t_pub/n:.2f} ms '
               f'(max {self.t_pub_max:.1f} · 콜백 밖)']
        if self.sink is not None:
            out.append(f'  sink 인계 {self.t_sink/n:.2f} ms (max {self.t_sink_max:.1f}) — frontend 예산과 분리 계상')
        return out
