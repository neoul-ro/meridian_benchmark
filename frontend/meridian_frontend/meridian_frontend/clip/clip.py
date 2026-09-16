#!/usr/bin/env python3
# clip.py — CLIP dual-output node (query_emb CLS + id_emb mask patch-pool). keyframe gate 호출 시에만 실행.
# 입력 = Preprocessing·Sam 인스턴스의 GPU 버퍼 직접 참조. TRT 실행 = 워커 스레드(S2 스트림).
# 출력 q/e 는 정규화되지 않은 엔진 원값. 소비자는 ev_emb 를 wait_event 한 뒤 읽고, L2 정규화는 publisher 가 수행.
import os
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')
import queue as _queue
import threading
import time
import tensorrt as trt
import torch
from torchvision.ops import roi_align


class Clip:
    def __init__(self, node, pre, sam, engine=None):
        self.node = node
        self.log = node.get_logger().get_child('clip')
        self.pre = pre
        self.sam = sam
        node.declare_parameter('clip_engine', engine)               # 경로는 frontend.engine_path 가 확정
        DEV = 'cuda'
        self.MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=DEV).view(1, 3, 1, 1)
        self.STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEV).view(1, 3, 1, 1)
        lg = trt.Logger(trt.Logger.ERROR)
        eng = trt.Runtime(lg).deserialize_cuda_engine(open(node.get_parameter('clip_engine').value, 'rb').read())
        self.ctx = eng.create_execution_context()
        self.S = torch.cuda.current_stream().cuda_stream
        self.q = torch.empty(self.sam.LANES, 512, device=DEV)  # 출력 슬롯, 엔진 주소 1회 바인딩
        self.e = torch.empty(self.sam.LANES, 512, device=DEV)
        self.roi_i = torch.zeros(self.sam.LANES, 5, device=DEV)              # [0|bbox] 상주, col0=0 고정
        self.roi_m = torch.zeros(self.sam.LANES, 5, device=DEV)              # [lane|bbox] 상주
        self.roi_m[:, 0] = torch.arange(self.sam.LANES, device=DEV, dtype=torch.float32)
        dummy_p = torch.zeros(8, 3, 224, 224, device=DEV)      # dyn-shape 웜업
        dummy_w = torch.ones(8, 49, device=DEV)
        self.ctx.set_input_shape('pixel_values', (8, 3, 224, 224)); self.ctx.set_input_shape('wpatch', (8, 49))
        self.ctx.set_tensor_address('pixel_values', dummy_p.data_ptr()); self.ctx.set_tensor_address('wpatch', dummy_w.data_ptr())
        self.ctx.set_tensor_address('query_emb', self.q.data_ptr()); self.ctx.set_tensor_address('id_emb', self.e.data_ptr())
        for _ in range(3):
            self.ctx.execute_async_v3(self.S)
        torch.cuda.synchronize()
        self.ev0 = torch.cuda.Event(enable_timing=True)        # engine GPU 시간, 다음 발화에서 수확
        self.ev1 = torch.cuda.Event(enable_timing=True)
        # 워밍업 이후 TRT ctx 는 워커 스레드만 사용
        self.S2 = torch.cuda.Stream()
        self.ev_in = torch.cuda.Event()                        # 제조(main) 완료 → 엔진(S2) 시작 경계
        self.ev_emb = torch.cuda.Event()                       # 엔진 완료 → 소비(main) 경계
        self._hold = None                                      # S2 소비 중 입력(pix·wp) 참조 보존
        self.p_roi = 0.0; self.p_trt = 0.0
        self._wq = _queue.Queue(maxsize=2)
        self.enq_done = threading.Event(); self.enq_done.set()
        self.worker_exc = None
        self.worker = threading.Thread(target=self._wloop, name='clip-trt-worker', daemon=True)
        self.worker.start()
        self.p_seq = None
        self.p_m = 0; self.p_cpu = 0.0
        self.m = 0                                             # m·lanes 는 publisher 가 읽음
        self.lanes = None
        self.seq_emb = -1
        self.n_fire = 0; self.n_emb = 0; self.gpu_ms = 0.0
        self.log.info('up — dual-output dyn-batch, 발화 판정 = tracker gate, 엔진 = 워커 스레드(S2)')

    def check_alive(self):
        """워커 비가동 시 예외 — 오래된 embedding 방출 방지."""
        if self.worker_exc is not None or not self.worker.is_alive():
            raise RuntimeError(f'clip 워커 스레드 비가동 — embedding 신뢰 불가 (원인: {self.worker_exc!r})')

    def _wloop(self):
        """워커 루프: 발화당 1건을 S2 에서 TRT enqueue. 예외는 worker_exc 에 기록."""
        try:
            while True:
                pix, wp, m = self._wq.get()
                with torch.cuda.stream(self.S2):
                    self.S2.wait_event(self.ev_in)
                    self.ctx.set_input_shape('pixel_values', (m, 3, 224, 224)); self.ctx.set_input_shape('wpatch', (m, 49))
                    self.ctx.set_tensor_address('pixel_values', pix.data_ptr()); self.ctx.set_tensor_address('wpatch', wp.data_ptr())
                    self.ev0.record()
                    self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
                    self.ev1.record()
                    self.ev_emb.record()
                pix.record_stream(self.S2); wp.record_stream(self.S2)   # allocator 재사용 보호
                self._hold = (pix, wp)
                self.enq_done.set()
                self._wq.task_done()
        except Exception as e:
            self.worker_exc = e
            self.log.error(f'clip 워커 스레드 사망: {e!r}')
            self.enq_done.set()

    def step(self, seq, det=None):
        """keyframe gate 발화 프레임에서만 호출. det = 인코딩할 레인 인덱스(long[m])."""
        self.check_alive()                                     # telemetry 수확보다 먼저
        if not self.enq_done.is_set():
            raise RuntimeError(f'seq={seq} clip 워커 직전 발화 enqueue 미완결 — 정체 상태, 새 발화 차단')
        if self.p_seq is not None:                             # 직전 발화 telemetry 수확
            self.ev1.synchronize()
            eng_ms = self.ev0.elapsed_time(self.ev1)
            self.gpu_ms += eng_ms
            self.log.info(f'seq={self.p_seq} n={self.p_m} 제조 cpu {self.p_cpu:.2f} '
                                   f'(roi/정규화 {self.p_roi:.2f} + trt발사 {self.p_trt:.2f}) + engine gpu {eng_ms:.2f} ms')
            self.p_seq = None
        t0 = self.node.get_clock().now()
        tp0 = time.perf_counter()
        idx = self.sam.det if det is None else det
        m = int(idx.shape[0])
        self.m = m; self.lanes = idx
        if m == 0:
            return
        bb = self.sam.bbox_p.index_select(0, idx)              # proto 좌표, 양쪽 roi 공유 (정합 조건)
        self.roi_i[:m, 1:] = bb
        self.roi_m[:m, 1:] = bb                                # col0 = arange, index_select 배치순과 일치
        imgn = ((self.pre.rgb_raw.permute(2, 0, 1)[None].float().div(255.0) - self.MEAN) / self.STD)   # 정규화→roi = roi→정규화 동치
        pix = roi_align(imgn, self.roi_i[:m], (224, 224), 2.5, aligned=True)
        wp = roi_align(self.sam.masks.index_select(0, idx).float().unsqueeze(1),
                       self.roi_m[:m], (7, 7), 1.0, aligned=True).reshape(m, 49).contiguous()
        tm1 = time.perf_counter()
        self.check_alive()
        self.ev_in.record()
        self.enq_done.clear()
        self._wq.put_nowait((pix, wp, m))                      # 큐 포화 = 워커 정체, 예외로 차단
        self.p_roi = (tm1 - tp0) * 1e3
        self.p_trt = (time.perf_counter() - tm1) * 1e3
        self.p_seq = seq; self.p_m = m
        self.seq_emb = seq
        self.n_fire += 1; self.n_emb += m
        self.p_cpu = (self.node.get_clock().now() - t0).nanoseconds / 1e6
