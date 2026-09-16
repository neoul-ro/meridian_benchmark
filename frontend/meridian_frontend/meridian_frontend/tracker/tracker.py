#!/usr/bin/env python3
"""Geometric tracker (IoU + centroid soft fusion) 와 keyframe gate.
score/apply/gate 는 CUDA graph 캡처. arena·입출력 버퍼는 주소 고정, grow 시 재캡처.
하류 계약: tid_cur[:N] 은 검출 lane 순서의 tracklet id · emb_tids/emb_pos = 이번 발화에 인코딩된 tracklet 과 det 내 위치.
"""
import time

import numpy as np
import torch
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo

DEV = 'cuda'

# association 상수
TRK_GATE3D = 0.30
TRK_PAT = 5
TRK_REVIVE = 10
TRK_SOFT_A0 = 422.0                                       # 배포 정본 (proto px, letterbox 등가 · research 500/175)
TRK_SOFT_S = 148.0
TRK_SOFT_TAU = 0.05
HASPOS_FLOOR = 10
TRK_MAXT = 128             # arena 용량. 초과 시 grow + graph 재캡처
TRK_COMPACT_EVERY = 30

# keyframe gate 상수
NOV_TAU = 0.20
KF_MIN_GAP = 5
KF_MIN_VALID = 0.10
BIRTH_MIN_AGE = 10
BIRTH_GAP = 10
NOV_ZMIN, NOV_ZMAX = 0.2, 5.0
GH_, GW_ = 48, 64


def greedy_assign(sc, N, T):
    """내림차순 flat 순회, first-come 1:1 배정."""
    out = -np.ones(N, dtype=np.int64)
    order = np.argsort(-sc, axis=None).tolist()
    flat = sc.ravel().tolist()
    det_used = bytearray(N); trk_used = bytearray(T)
    for bi in order:
        if flat[bi] <= -0.5:
            break
        r = bi // T; c = bi - r * T
        if det_used[r] or trk_used[c]:
            continue
        det_used[r] = 1; trk_used[c] = 1; out[r] = c
    return out


class Tracker:
    def __init__(self, node, pre, sam, clip=None, info_topic=None, nov_tau=None):
        self.node = node
        self.log = node.get_logger().get_child('tracker')
        self.nov_tau = NOV_TAU if nov_tau is None else float(nov_tau)
        self.n_enc = 0
        self.pre, self.sam, self.clip = pre, sam, clip
        self.PH, self.PW = sam.PH, sam.PW
        self.L = sam.LANES
        self.P = self.PH * self.PW
        self.a0, self.s = TRK_SOFT_A0, TRK_SOFT_S              # graph 캡처 상수. 런타임 변경 불가
        self.count = -1
        self.cap = TRK_MAXT
        self.next_id = 0
        self.CH = 2048                                         # fp16 정수 정확 상한 (2^11)
        self.NC = self.P // self.CH
        assert self.NC * self.CH == self.P

        # CPU 장부 메타데이터
        c = self.cap
        self.m_tid = np.zeros(c, np.int64)
        self.m_last = np.zeros(c, np.int64)
        self.m_born = np.zeros(c, np.int64)
        self.m_alive = np.zeros(c, bool)
        self.m_embedded = np.zeros(c, bool)
        self.m_haspos = np.zeros(c, bool)
        self.m_nobs = np.zeros(c, np.int64)
        self.n = 0

        # GPU payload arena. graph 캡처 주소, 재바인딩 금지
        self._alloc_arenas(self.cap)

        # graph 입출력 고정 버퍼
        self.cbuf = torch.zeros(self.L, 3, device=DEV)         # sam.centroid 는 매 프레임 새 텐서, 제자리 복사
        self.msrc16c = torch.zeros(self.NC, self.L, self.CH, dtype=torch.float16, device=DEV)  # mask fp16, 청크-major
        self.area_g = torch.zeros(self.L, device=DEV)
        self.colmask = torch.zeros(self.cap, dtype=torch.bool, device=DEV)
        self.colmask_pin = torch.zeros(self.cap, dtype=torch.bool, pin_memory=True)
        self.colmask_np = self.colmask_pin.numpy()
        self.neg1 = torch.tensor(-1.0, device=DEV)
        self.zero_s = torch.zeros((), device=DEV)
        self.ghost_g = torch.zeros((), dtype=torch.long, device=DEV)
        self.sam_masks_flat = self.sam.masks.view(self.L, self.P)
        self._alloc_out(self.cap)

        # apply 스테이징. pinned 고정 주소, graph 내 memcpy
        self.idx_pin = torch.zeros(4 * self.L, dtype=torch.int64, pin_memory=True)
        self.idx_np = self.idx_pin.numpy()
        self.idx_g = torch.zeros(4 * self.L, dtype=torch.int64, device=DEV)
        self.pf_pin = torch.zeros(self.L, dtype=torch.float32, pin_memory=True)
        self.pf_np = self.pf_pin.numpy()
        self.pf_g = torch.zeros(self.L, dtype=torch.float32, device=DEV)
        self.tid_pin = torch.zeros(self.L, dtype=torch.int64, pin_memory=True)
        self.tid_np = self.tid_pin.numpy()
        self.tid_full = torch.zeros(self.L, dtype=torch.int64, device=DEV)
        self.tid_cur = self.tid_full[:0]
        self.is_kf = False; self.kf_reason = ''
        self.emb_lanes = None

        # gate: coverage novelty
        Ho, Wo = 480, 640
        self.Ho, self.Wo = Ho, Wo
        self.GVI = torch.linspace(0, Ho - 1, GH_, device=DEV).long()
        self.GUI = torch.linspace(0, Wo - 1, GW_, device=DEV).long()
        self.GV = self.GVI.float().view(GH_, 1).expand(GH_, GW_).reshape(-1)
        self.GU = self.GUI.float().view(1, GW_).expand(GH_, GW_).reshape(-1)
        self.Ko = torch.tensor([415.692, 415.692, 320.0, 240.0], device=DEV)
        self._K_latched = False
        self.kf_R = torch.eye(3, device=DEV)
        self.kf_t = torch.zeros(3, device=DEV)
        self.kf_depth = torch.zeros(Ho, Wo, device=DEV)
        self.anchored = False
        self.nov_graph = None; self.g_out = None
        self.gate_pin = torch.empty(2, dtype=torch.float32, pin_memory=True)   # [novelty, valid frac]
        self.ev_gate = torch.cuda.Event()
        self.have_gate = False
        self.last_kf = -10 ** 9; self.last_birth = -10 ** 9

        # 통계
        self.hist_np = np.zeros(512, np.int64)
        self.revived = 0
        self.nopos = 0
        self.cov = np.zeros(4, np.int64)                       # [den8, num8, den3, num3]
        self.n_kf = 0; self.n_cov = 0; self.n_birth = 0
        self.n_obs = 0; self.n_frame = 0
        self.nov_log = []
        self.t_up = 0.0; self.t_sync = 0.0; self.t_greedy = 0.0; self.t_apply = 0.0; self.t_gate = 0.0

        self.ev = torch.cuda.Event()
        self._info_sub = node.create_subscription(
            CameraInfo, info_topic or '/camera/color/camera_info', self.on_info,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.score_graph = None
        self.apply_graph = None
        self._capture_score()
        self._capture_apply()
        self._capture_gate()
        self.log.info(
            f'up(opt) — soft-fusion a0={self.a0:.0f} s={self.s:.0f} tau={TRK_SOFT_TAU} gate3d={TRK_GATE3D} '
            f'patience={TRK_PAT} cap={self.cap} | score/apply=CUDA graph(fp16 청크 bmm 정확)+CPU 장부 | '
            f'gate: coverage tau={self.nov_tau} + birth age>={BIRTH_MIN_AGE}')

    # ---------------- 버퍼 ----------------
    # fp16 청크 bmm: 청크 부분합 ≤ CH 이므로 fp16 정확
    # 끝 +1 행 = apply 패딩 scratch, score 는 [:cap] 만 읽음
    def _alloc_arenas(self, cap):
        self.tmask16 = torch.zeros(self.NC, cap + 1, self.CH, dtype=torch.float16, device=DEV)
        self.tarea = torch.zeros(cap + 1, device=DEV)
        self.tpos = torch.zeros(cap + 1, 3, device=DEV)
        self.thaspos_f = torch.zeros(cap + 1, device=DEV)      # 0/1 f32, graph 내 >0.5 판독

    def _alloc_out(self, cap):
        R = self.L
        sz = 2 * R * cap + 3 * R
        self.gout = torch.zeros(sz, device=DEV)
        self.gs = self.gout[:R * cap].view(R, cap)
        self.gd = self.gout[R * cap:2 * R * cap].view(R, cap)
        self.gh = self.gout[2 * R * cap:2 * R * cap + R]
        self.ga = self.gout[2 * R * cap + R:2 * R * cap + 2 * R]
        self.gv = self.gout[2 * R * cap + 2 * R:]
        self.gpin = torch.empty(sz, dtype=torch.float32, pin_memory=True)
        gp = self.gpin.numpy()
        self.np_score = gp[:R * cap].reshape(R, cap)
        self.np_d = gp[R * cap:2 * R * cap].reshape(R, cap)
        self.np_hascur = gp[2 * R * cap:2 * R * cap + R]
        self.np_area = gp[2 * R * cap + R:2 * R * cap + 2 * R]
        self.np_valid = gp[2 * R * cap + 2 * R:]

    # ---------------- score graph ----------------
    def _score_core(self):
        """패딩 전쌍 점수 계산. colmask H2D·gout D2H 포함 전부 graph 노드."""
        self.colmask.copy_(self.colmask_pin, non_blocking=True)   # alive 슬롯 마스크 H2D
        self.msrc16c.copy_(self.sam_masks_flat.view(self.L, self.NC, self.CH).permute(1, 0, 2))   # bool→fp16 청크-major
        self.area_g.copy_(self.sam.area)
        hascur = self.sam.counts > HASPOS_FLOOR
        inter = torch.bmm(self.msrc16c, self.tmask16[:, :self.cap, :].transpose(1, 2)).float().sum(0)   # 청크 부분합 ≤ CH, fp16 정확
        union = self.area_g[:, None] + self.tarea[:self.cap][None, :] - inter
        iou = inter / union.clamp(min=1e-6)
        thas = self.thaspos_f[:self.cap] > 0.5
        both = hascur[:, None] & thas[None, :]
        diff = self.cbuf[:, None, :] - self.tpos[:self.cap][None, :, :]
        d = diff.pow(2).sum(-1).sqrt()                         # cdist 대신 차분식 (shape 무관 경로)
        within = both & (d <= TRK_GATE3D)
        w = torch.sigmoid((self.area_g - self.a0) / self.s)
        cen_raw = torch.clamp(1.0 - d / TRK_GATE3D, min=0.0)
        cen = torch.where(both, cen_raw, self.zero_s)
        pairv = self.sam.valid[:, None] & self.colmask[None, :]
        self.ghost_g += (pairv & (~both) & (cen_raw > 0.0) & (iou > 0.0)).sum()
        fused = w[:, None] * (1.0 + iou) + (1.0 - w[:, None]) * cen
        veto = both & (d > TRK_GATE3D)
        adm = (~veto) & ((iou > 0.0) | (within & (cen > 0.0))) & (fused >= TRK_SOFT_TAU)
        self.gs.copy_(torch.where(adm & pairv, fused, self.neg1))
        self.gd.copy_(d)
        self.gh.copy_(hascur.float())
        self.ga.copy_(self.area_g)
        self.gv.copy_(self.sam.valid.float())
        self.gpin.copy_(self.gout, non_blocking=True)

    def _capture_score(self):
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                self._score_core()
        torch.cuda.current_stream().wait_stream(st)
        torch.cuda.synchronize()
        self.score_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.score_graph):
            self._score_core()
        torch.cuda.synchronize()
        self.ghost_g.zero_()                                   # 웜업 잔여 오염 제거

    # ---------------- apply graph ----------------
    def _apply_core(self):
        """고정 L 슬롯 payload 반영. idx_g = [mask 행 | lane | pos 행 | pos lane], 패딩은 scratch 행 cap."""
        L = self.L
        self.idx_g.copy_(self.idx_pin, non_blocking=True)
        self.pf_g.copy_(self.pf_pin, non_blocking=True)
        self.tid_full.copy_(self.tid_pin, non_blocking=True)
        a_rows = self.idx_g[:L]; a_lanes = self.idx_g[L:2 * L]
        p_rows = self.idx_g[2 * L:3 * L]; p_lanes = self.idx_g[3 * L:4 * L]
        self.tmask16.index_copy_(1, a_rows, self.msrc16c.index_select(1, a_lanes))
        self.tarea.index_copy_(0, a_rows, self.area_g.index_select(0, a_lanes))
        self.tpos.index_copy_(0, p_rows, self.cbuf.index_select(0, p_lanes) * self.pf_g[:, None])
        self.thaspos_f.index_copy_(0, p_rows, self.pf_g)

    def _capture_apply(self):
        self.idx_np[:self.L] = self.cap                        # 캡처 입력 = 전 슬롯 scratch
        self.idx_np[self.L:2 * self.L] = 0
        self.idx_np[2 * self.L:3 * self.L] = self.cap
        self.idx_np[3 * self.L:] = 0
        self.pf_np[:] = 0.0
        self.idx_g.copy_(self.idx_pin); self.pf_g.copy_(self.pf_pin)
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                self._apply_core()
        torch.cuda.current_stream().wait_stream(st)
        torch.cuda.synchronize()
        self.apply_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.apply_graph):
            self._apply_core()
        torch.cuda.synchronize()

    def _grow(self, need):
        newcap = max(self.cap * 2, need)
        self.log.warn(f'arena grow {self.cap} → {newcap} — score/apply graph 재캡처')
        old = (self.tmask16, self.tarea, self.tpos, self.thaspos_f)
        n = self.n
        oldcap = self.cap
        self.cap = newcap
        self._alloc_arenas(newcap)
        self.tmask16[:, :n, :] = old[0][:, :n, :]; self.tarea[:n] = old[1][:n]
        self.tpos[:n] = old[2][:n]; self.thaspos_f[:n] = old[3][:n]

        def gnp(a):
            z = np.zeros(newcap, a.dtype); z[:n] = a[:n]; return z
        self.m_tid = gnp(self.m_tid); self.m_last = gnp(self.m_last); self.m_born = gnp(self.m_born)
        self.m_alive = gnp(self.m_alive); self.m_embedded = gnp(self.m_embedded)
        self.m_haspos = gnp(self.m_haspos); self.m_nobs = gnp(self.m_nobs)
        self.colmask = torch.zeros(newcap, dtype=torch.bool, device=DEV)
        self.colmask_pin = torch.zeros(newcap, dtype=torch.bool, pin_memory=True)
        self.colmask_np = self.colmask_pin.numpy()
        self.cap = newcap
        self._alloc_out(newcap)
        self._capture_score()
        self._capture_apply()

    # ---------------- gate ----------------
    def on_info(self, m):
        if self._K_latched:
            return
        self.Ko.copy_(torch.tensor([m.k[0], m.k[4], m.k[2], m.k[5]]))
        self._K_latched = True
        if self._info_sub is not None:
            self.node.destroy_subscription(self._info_sub); self._info_sub = None
        self.log.info(f'gate intrinsics latch — fx={m.k[0]:.2f} fy={m.k[4]:.2f} cx={m.k[2]:.2f} cy={m.k[5]:.2f}')

    def _novelty_core(self):
        d = self.pre.depth_m[self.GVI][:, self.GUI].reshape(-1)
        fx, fy, cx, cy = self.Ko[0], self.Ko[1], self.Ko[2], self.Ko[3]
        validc = (d > NOV_ZMIN) & (d < NOV_ZMAX)
        cam = torch.stack([(self.GU - cx) / fx * d, (self.GV - cy) / fy * d, d], 1)
        world = cam @ self.sam.Rw.T + self.sam.tw
        camp = (world - self.kf_t) @ self.kf_R
        zp = camp[:, 2]
        up = fx * camp[:, 0] / zp.clamp(min=1e-3) + cx
        vp = fy * camp[:, 1] / zp.clamp(min=1e-3) + cy
        inb = (zp > NOV_ZMIN) & (zp < NOV_ZMAX) & (up >= 0) & (up < self.Wo) & (vp >= 0) & (vp < self.Ho)
        ui = up.clamp(0, self.Wo - 1).long(); vi = vp.clamp(0, self.Ho - 1).long()
        d_old = self.kf_depth[vi, ui]
        d_ok = (d_old > NOV_ZMIN) & (d_old < NOV_ZMAX)
        seen = inb & d_ok & (torch.abs(d_old - zp) <= 0.10 * zp)
        unknown = inb & (~d_ok)
        denom = (validc & (~unknown)).sum().float()
        new = (validc & (~seen) & (~unknown)).sum().float()
        return torch.stack([new / denom.clamp(min=1.0), denom / float(GH_ * GW_)])

    def _capture_gate(self):
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(5):
                self._novelty_core()
        torch.cuda.current_stream().wait_stream(st)
        torch.cuda.synchronize()
        self.nov_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.nov_graph):
            self.g_out = self._novelty_core()
        torch.cuda.synchronize()

    def _anchor(self):
        self.kf_depth.copy_(self.pre.depth_m)
        self.kf_R.copy_(self.sam.Rw); self.kf_t.copy_(self.sam.tw)
        self.anchored = True

    # ---------------- 판정 ----------------
    def _decide(self):
        if not self.anchored:
            self.last_kf = self.count; self.n_kf += 1; self.n_cov += 1
            return True, 'init'
        if not self.have_gate:
            return False, ''
        self.ev_gate.synchronize()
        nov, vfrac = self.gate_pin.tolist()
        self.nov_log.append(nov)
        c = self.count - 1                                     # gate 값은 직전 프레임(count-1) 기준
        birth = False
        if self.n:
            birth = bool(np.any(self.m_alive[:self.n] & (~self.m_embedded[:self.n])
                                & (self.m_last[:self.n] >= c)
                                & ((c - self.m_born[:self.n]) >= BIRTH_MIN_AGE)))
        cov_fire = (nov > self.nov_tau) and (vfrac >= KF_MIN_VALID)
        birth_fire = (not cov_fire) and birth and (self.count - self.last_birth >= BIRTH_GAP)
        is_kf = (cov_fire or birth_fire) and (self.count - self.last_kf >= KF_MIN_GAP)
        if is_kf:
            self.last_kf = self.count; self.n_kf += 1
            if cov_fire:
                self.n_cov += 1
            else:
                self.last_birth = self.count; self.n_birth += 1
        return is_kf, ('cov' if cov_fire else 'birth') if is_kf else ''

    # ---------------- 장부 ----------------
    def _retire(self):
        if self.n:
            self.m_alive[:self.n] &= ~((self.count - self.m_last[:self.n]) > TRK_PAT)

    def _compact(self, force=False):
        if not ((self.count % TRK_COMPACT_EVERY) == 0 or self.n > self.cap - 16 or force):
            return
        if self.n == 0:
            return
        keep = self.m_alive[:self.n] | ((self.count - self.m_last[:self.n]) <= TRK_REVIVE)
        idx = np.flatnonzero(keep); m = len(idx)
        if m == self.n:
            return
        drop = np.flatnonzero(~keep)
        self._harvest(drop)
        for a in (self.m_tid, self.m_last, self.m_born, self.m_nobs):
            a[:m] = a[idx]
        for a in (self.m_alive, self.m_embedded, self.m_haspos):
            a[:m] = a[idx]
        self.m_alive[m:self.n] = False
        gi = torch.as_tensor(idx, device=DEV)
        self.tmask16[:, :m, :] = self.tmask16.index_select(1, gi)
        self.tarea[:m] = self.tarea.index_select(0, gi)
        self.tpos[:m] = self.tpos.index_select(0, gi)
        self.thaspos_f[:m] = self.thaspos_f.index_select(0, gi)
        self.n = m

    def _harvest(self, rows):
        if len(rows) == 0:
            return
        nb = self.m_nobs[rows]; em = self.m_embedded[rows]
        d8 = nb >= 8; d3 = nb >= 3
        self.cov += np.array([d8.sum(), (d8 & em).sum(), d3.sum(), (d3 & em).sum()], np.int64)

    # ---------------- 프레임 ----------------
    def step(self, seq, stamp):
        self.count += 1
        self.is_kf, self.kf_reason = self._decide()
        if self.is_kf:
            # CLIP enqueue 전에 추출 (동기점 순서)
            self.sam.extract_points()
        N = int(self.sam.det.numel())
        tidc = np.zeros(0, np.int64)
        if self.n + N > self.cap:                              # 전 검출 birth 대비 사전 확보
            self._compact(force=True)
            if self.n + N > self.cap:
                self._grow(self.n + N)
        if N > 0:
            t0 = time.perf_counter()
            # graph 입력 주입 + replay
            self.cbuf.copy_(self.sam.centroid)
            self.colmask_np[:] = False
            self.colmask_np[:self.n] = self.m_alive[:self.n]
            self.score_graph.replay()
            self.ev.record()
            t1 = time.perf_counter(); self.t_up += (t1 - t0) * 1e3
            self.ev.synchronize()
            t2 = time.perf_counter(); self.t_sync += (t2 - t1) * 1e3
            # greedy (유효 lane × alive 행 압축 좌표)
            lanes = np.flatnonzero(self.np_valid > 0.5)
            hascur = self.np_hascur > 0.5
            acts = np.flatnonzero(self.m_alive[:self.n]); T = len(acts)
            assign = None
            if T > 0:
                assign = greedy_assign(self.np_score[np.ix_(lanes, acts)], N, T)
            t3 = time.perf_counter(); self.t_greedy += (t3 - t2) * 1e3
            # 장부 갱신
            tidc = np.full(N, -1, np.int64)
            grow = lanes_m = growp = lanesp = np.zeros(0, np.int64)
            if assign is not None and (assign >= 0).any():
                mrows = np.nonzero(assign >= 0)[0]
                lanes_m = lanes[mrows]
                grow = acts[assign[mrows]]
                self.m_last[grow] = self.count
                self.m_nobs[grow] += 1
                tidc[mrows] = self.m_tid[grow]
                hc = hascur[lanes_m]
                growp = grow[hc]; lanesp = lanes_m[hc]
                self.m_haspos[growp] = True
            u = np.nonzero(tidc < 0)[0]
            s0 = e0 = self.n
            uhas = np.zeros(0, bool)
            if len(u):
                lanes_u = lanes[u]
                uhas = hascur[lanes_u]
                dead = np.flatnonzero(~self.m_alive[:self.n]
                                      & ((self.count - self.m_last[:self.n]) <= TRK_REVIVE)
                                      & self.m_haspos[:self.n])
                if len(dead):
                    dd = self.np_d[np.ix_(lanes_u, dead)]
                    self.revived += int((uhas & (dd <= TRK_GATE3D).any(1)).sum())
                K = len(u)
                s0 = self.n; e0 = s0 + K
                ids = np.arange(self.next_id, self.next_id + K, dtype=np.int64)
                self.m_tid[s0:e0] = ids
                self.m_last[s0:e0] = self.count
                self.m_born[s0:e0] = self.count
                self.m_alive[s0:e0] = True
                self.m_embedded[s0:e0] = False
                self.m_nobs[s0:e0] = 1
                self.m_haspos[s0:e0] = uhas
                tidc[u] = ids
                self.n = e0; self.next_id += K
            # apply 스테이징 (compact 이전 인덱스, 패딩 = scratch 행)
            nm = len(grow); nb = e0 - s0
            npw = len(growp) + nb
            L = self.L
            self.idx_np[:L] = self.cap
            self.idx_np[:nm] = grow
            self.idx_np[nm:nm + nb] = np.arange(s0, e0)
            self.idx_np[L:2 * L] = 0
            self.idx_np[L:L + nm] = lanes_m
            if nb:
                self.idx_np[L + nm:L + nm + nb] = lanes[u]
            self.idx_np[2 * L:3 * L] = self.cap
            self.idx_np[2 * L:2 * L + len(growp)] = growp
            self.idx_np[2 * L + len(growp):2 * L + npw] = np.arange(s0, e0)
            self.idx_np[3 * L:4 * L] = 0
            self.idx_np[3 * L:3 * L + len(growp)] = lanesp
            if nb:
                self.idx_np[3 * L + len(growp):3 * L + npw] = lanes[u]
            self.pf_np[:] = 0.0
            self.pf_np[:len(growp)] = 1.0
            self.pf_np[len(growp):npw] = uhas.astype(np.float32)
            self.tid_np[:N] = tidc
            self.apply_graph.replay()
            self.tid_cur = self.tid_full[:N]
            # 통계
            self.hist_np += np.bincount(
                (self.np_area[lanes] / np.float32(128.0)).astype(np.int64).clip(0, 511), minlength=512)
            self.nopos += int((~hascur[lanes]).sum())
            self._retire()
            self._compact()
            self.t_apply += (time.perf_counter() - t3) * 1e3
        else:
            self.tid_cur = self.tid_full[:0]
            self._retire()
            self._compact()
        # 발화 시 CLIP. embedded 표시는 tid 기준 (compact 후)
        self.emb_lanes = None
        self.emb_tids = None
        self.emb_pos = None
        if self.is_kf and self.clip is not None and N > 0:
            det = self.sam.det                                 # cov/init = 전 검출
            enc_tids = tidc
            enc_pos = np.arange(N)
            # birth = 미임베딩 tracklet 만 인코딩
            if self.kf_reason == 'birth' and self.n:
                emb_tids = self.m_tid[:self.n][self.m_embedded[:self.n]]
                sel = np.flatnonzero(~np.isin(tidc, emb_tids))
                det = self.sam.det[torch.from_numpy(sel).to(DEV)] if sel.shape[0] else None
                enc_tids = tidc[sel]
                enc_pos = sel
            if det is not None:
                self.clip.step(seq, det)
                self.emb_lanes = det
                self.emb_tids = enc_tids                       # embedding 순서 = 이 tracklet 순서
                self.emb_pos = enc_pos                         # det 순서 내 위치 · publisher 가 N행에 배치
                self.n_enc += int(det.numel())
                if self.n:
                    self.m_embedded[:self.n] |= np.isin(self.m_tid[:self.n], enc_tids)
        # anchor ratchet (cov/init 만)
        t4 = time.perf_counter()
        if self.kf_reason in ('cov', 'init'):
            self._anchor()
        # 다음 프레임용 novelty, enqueue만
        self.nov_graph.replay()
        self.gate_pin.copy_(self.g_out, non_blocking=True)
        self.ev_gate.record()
        self.have_gate = True
        self.t_gate += (time.perf_counter() - t4) * 1e3
        self.n_obs += N; self.n_frame += 1
        if self.n_frame % 100 == 0:
            self.log.info(
                f'seq={seq} 관측 {N} · tracklet 누적 {self.next_id} · alive {int(self.m_alive[:self.n].sum())} | '
                f'assoc {(self.t_up + self.t_sync + self.t_greedy + self.t_apply)/100:.2f} ms '
                f'(up {self.t_up/100:.2f} · sync {self.t_sync/100:.2f} · greedy {self.t_greedy/100:.2f} · '
                f'반영 {self.t_apply/100:.2f}) · gate {self.t_gate/100:.2f} ms | '
                f'kf {self.n_kf} (cov {self.n_cov} birth {self.n_birth}) '
                f'enc {self.n_enc} tau={self.nov_tau}')
            self.t_up = self.t_sync = self.t_greedy = self.t_apply = self.t_gate = 0.0

    def summary(self):
        if self.n:
            self._harvest(np.arange(self.n))
        cov = self.cov
        h = self.hist_np
        tot = h.sum()
        lines = [f'tracker: 프레임 {self.n_frame} · 관측 {self.n_obs} · tracklet 발급 {self.next_id} '
                 f'(관측/tracklet {self.n_obs / max(self.next_id, 1):.2f})',
                 f'  revived(ID switch proxy) {self.revived} · ghost-cen guard 발화 {int(self.ghost_g.item())} '
                 f'· 위치없는 검출 {self.nopos} ({100.0 * self.nopos / max(self.n_obs, 1):.2f}%)',
                 f'  keyframe {self.n_kf} / {self.n_frame} 프레임 ({100.0*self.n_kf/max(self.n_frame,1):.1f}%, '
                 f'cov {self.n_cov} · birth {self.n_birth})',
                 f'  emb-coverage ≥8obs {100.0*cov[1]/max(cov[0],1):.1f}% ({cov[1]}/{cov[0]}) · '
                 f'≥3obs {100.0*cov[3]/max(cov[2],1):.1f}% ({cov[3]}/{cov[2]})']
        if tot > 0:
            c = np.cumsum(h) / tot
            for q in (50, 90, 95):
                b = int(np.searchsorted(c, q / 100.0))
                lines.append(f'  detection area p{q} ≈ {b * 128 + 64} proto-px  (a0 앵커 {self.a0:.0f} s {self.s:.0f})')
        if self.nov_log:
            v = np.array(self.nov_log)
            lines.append(f'  novelty median {np.median(v):.3f} · p90 {np.percentile(v, 90):.3f} '
                         f'(tau {self.nov_tau})')
        return lines
