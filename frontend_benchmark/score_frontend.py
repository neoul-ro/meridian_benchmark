#!/usr/bin/env python3
"""frontend 출력(run_frontend.py) 을 uHumans2 GT 로 채점한다. 위키 Workspace/Evaluation TODO 1 의 두 항목.

  (1) 놓친 것  — 물체가 있었는데 감지하지 못한 것
  (2) 위치 정확도 — 탐지한 물체를 복셀화해서 실제 맵의 물체와 겹치는 정도 · 거리 오차가 20cm 이내인지

3D 매칭 기준 (26/09/17 사용자 결정 ④ — 규칙 본문·출처는 match3d.py docstring 한 곳)
  예전: share = |τ 안 예측 복셀| / |예측 복셀| (intersection over prediction) 최대 헝가리안 → 작은 순수 조각을 선호했다
        (apartment kf29 / ep78: share 1.0 인 192복셀 조각이 share 0.87 인 2403복셀 조각을 이김, completeness 1.9 → 71cm).
  지금: keyframe 마다 그 keyframe 에 보인 GT 표면과의 허용 거리 IoU_τ ≥ 0.5 헝가리안 (MOTS · ScanNet 인스턴스 IoU ·
        TrackEval 유사도 · 2D PQ 와 같은 발상). score_mot 도 같은 IoU_τ 를 유사도로 쓴다.

GT 표면   gt_surface.LabelSurface — keyframe 의 2D GT 라벨(값 ep+1) 픽셀을 depth·K·GT pose 로 역투영, 5m 이내, 2cm 복셀
          (score_geometry 의 keyframe GT 와 복셀 단위로 같다). 라벨 폴더 = labels_dir (CLI --labels), meta.json version·sha1 을
          params 에 남긴다. 후보 GT = 그 keyframe 에 표면 복셀이 하나라도 있는 GT 전부 (작은 것 포함).
present   그 keyframe 의 라벨 픽셀(5m 이내·depth 유효) ≥ area_min_px(1600 = sam.AREA_MIN 256 proto 칸 × 2.5²).
          present 아닌 GT 에 매칭된 예측 = 제외 매칭(ignored match): TP 도 FP 도 아니다 (detect_credit = 0).
예측      관측의 점 중 카메라 max_range(5m) 이내 → 2cm 복셀.
IoU_τ     P_τ·R_τ/(P_τ+R_τ−P_τ·R_τ), d < τ strict. τ = 10·20·50cm 각각, 주 τ = 20cm.
매칭      τ 마다 IoU_τ ≥ 0.5 (TrackEval clear.py:82 '≥ thr − eps') 쌍, 헝가리안 IoU 합 최대. 시간 정보 없음(감사 A-4) —
          시간 연관은 score_mot.py 의 CLEAR 매칭이 맡는다.
미매칭 예측 FP / 제외  (void + present 아닌 GT 위 복셀)/|P| > 0.5 → 제외, 아니면 FP (panopticapi). 이유를 excluded_reason 에.
중복      dup_of = TIDE Dupe (수정 ⑤ E2): 매칭 안 된 예측이 같은 keyframe 에서 다른 예측에 매칭된 present GT 와 IoU_τ ≥ 0.5
          (Loc 범위 [0.1, 0.5] 이면 Loc 이 먼저 — quantify.py:236-255). MOT 에서는 FP. 매칭 안 된 예측마다 TIDE 오류
          (dupe · loc · bkg · other) 를 tide_error 열에 — FP / 제외 판정과는 따로 (규칙 본문 tide_rules.py)
탐지(트랙) 채점 대상(eligible, gt_vis 의 crop px 최대 ≥ 1600 — 그대로) 에피소드가 present 인 keyframe 에서 1회 이상 매칭되면 탐지.
원인      present 인데 매칭 안 된 (keyframe, GT) 마다 merged / split / low_iou / missed (score_2d 규칙, match3d).
          merged · split 의 덮음은 가장 가까운 GT 소유로 배타적 (수정 ⑤ E1) · low_iou = 최고 IoU_τ ∈ [0.1, 0.5] (2D 와 같은 규칙, E3 · F2 TIDE 기본 0.1).
          트랙: present keyframe 없음 → no_kf, 아니면 최빈 상태 (동률: merged > split > low_iou > missed).
          옛 결과(26/09/17 이전)의 원인 값 대응은 params.reason_mapping.
위치 정확도 매칭된 관측마다(제외 매칭 포함, detect_credit 로 구분) 예측 복셀 → 매칭된 에피소드 점군(구간 전체) 최근접 거리
          · 거리 오차 = 중앙값/평균/p90 (cm) · within_tau = P@τ (d < τ) · voxel_exact = 같은 2cm 셀 비율
          · coverage_at_tau = R@τ: 에피소드마다 탐지 인정된 관측 복셀 합집합으로 에피소드 점군을 τ 안에 덮은 비율
민감도    tau = 10/20/50cm 로 바꿨을 때 재현율·매칭률
부분 실행 frontend 를 [fr_lo, fr_hi) 만 돌렸으면 그 구간에 걸친 에피소드만, 가시성도 그 구간 프레임만 센다 (감사 A-8)
빈 입력   분모가 0 인 비율은 null (NaN 을 JSON 에 쓰지 않는다 — RFC 8259, 감사 A-9)

출력 (기존 열·키는 이름·위치 유지, 새 것은 뒤에)
  score_observations.csv  관측 1행. 새 열 gt_vox@20 (에피소드별 가장 가까운 표면 복셀 수) · gt_iou@20 ('ep:IoU' IoU>0 전부,
                          17 유효숫자 — score_mot 유사도) · match_iou/P/R@20 · pred_status(tp·ignored_match·fp·excluded) ·
                          excluded_reason · void_frac@20 · ignore_frac@20 · dup_of_ep
  score_kf_gt.csv         (keyframe, 표면 있는 GT) 1행 — present · 상태(tp·merged·split·low_iou·missed·ignored_match·not_present) ·
                          최고 IoU · 매칭 관측. score_mot 이 present 를 여기서 읽는다.
                          n_frags · split_cover 는 배타 덮음 기준 (수정 ⑤ E1)
  score_observations.csv  수정 ⑤ 새 열 tide_error (매칭 안 된 예측의 TIDE 오류, 매칭된 예측은 빈 값)
  score_episodes.csv      에피소드 1행. 새 열 n_kf_with_surface · n_kf_present · n_kf_<상태> · best_iou(present keyframe 에서의
                          최고 IoU_τ, present keyframe 이 없으면 빈 값)
  score.json              새 블록 keyframe_gt (present 쌍 · 상태 · 라벨/가시성 present 불일치)
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import WS, MODELS  # noqa: E402,F401  (meridian_benchmark 경로도 여기서 잡힘)
from meridian_benchmark.uhumans2 import UH2Sequence  # noqa: E402
import gt_surface as GS  # noqa: E402
import match3d as M  # noqa: E402

EPS = M.EPS          # 거리 여유 (m). strict d < τ − EPS — 근거는 match3d.py docstring '거리 판정' (옛 d ≤ τ + EPS 는 폐지)
NONE = 0             # GT 물체 번호는 1부터 (확인: gt_object_id 최소값 검사)
# 사람 prefab 색. 사람 0명 시퀀스(00h)에는 없고 사람 있는 시퀀스에만 나오는 유일한 색(office·apartment 공통),
# 그 에피소드 높이 중앙값 1.5~1.6m. 움직이는 사람은 3D 겹침이 없어 GT 에서 여러 물체·에피소드로 쪼개져 있다.
HUMAN_COLORS = ('23d5ea',)
KF_STATUSES = ('tp', 'merged', 'split', 'low_iou', 'missed', 'ignored_match', 'not_present')
SIMILARITY = ('tolerant IoU_tau = P*R/(P+R-P*R), P = pred voxels within tau of GT surface / |pred|, '
              'R = GT surface voxels within tau of pred / |GT| (d < tau strict)')      # report.py 가 기대값으로 읽는다
TIDE_ERRORS = ('dupe', 'loc', 'bkg', 'other')


def tau_cm(t):
    """열 이름의 τ 표기 (match@20 · gt_counts@20 · gt_iou@20). score_mot 도 score.json params 에서 같은 이름을 읽는다."""
    return int(round(t * 100))


def voxelize(p, v):
    """world 점 → 셀 중심 (GT 와 같은 규칙: (floor(p/v)+0.5)·v), 중복 제거. gt_surface.voxelize 와 같다."""
    return GS.voxelize(p, v)


def vkeys(c, v):
    """셀 중심 → 정수 셀 키 (1D int64 로 접어서 집합 연산)."""
    k = np.floor(c / v).astype(np.int64) + (1 << 20)
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


def format_counts(counter):
    """{키: 복셀 수} → 'key:count;key:count' (키 오름차순, 0개는 없음)."""
    return ';'.join(f'{int(o)}:{int(n)}' for o, n in sorted(counter.items()) if n > 0)


def parse_counts(s):
    """format_counts 의 역. 빈 문자열 → {}."""
    return {int(a): int(b) for a, b in (x.split(':') for x in s.split(';') if x)} if s else {}


class GT:
    def __init__(self, path):
        self.f = h5py.File(path, 'r')
        m = self.f['_metadata']
        self.voxel = float(m['voxel_m'][()]); self.max_range = float(m['max_range_m'][()])
        self.gap = int(m['gap_frames'][()])
        ix = self.f['index']
        self.tid = ix['tracklet_id'][:]; self.oid = ix['gt_object_id'][:]
        self.first = ix['first_frame'][:]; self.last = ix['last_frame'][:]
        self.nfo = ix['n_frames_observed'][:]; self.npts = ix['n_points'][:]
        self.dist_min = ix['dist_min_m'][:]
        self.flushed = ix['flushed_at_end'][:]
        if len(self.oid) and self.oid.min() <= NONE:
            raise SystemExit('gt_object_id 가 0 이하 — NONE 라벨과 충돌')
        cols = [self.f[f'tracklets/{int(t):05d}/_metadata/color_hex'][()] for t in self.tid]
        self.color = np.array([c.decode() if isinstance(c, bytes) else c for c in cols])
        self._pts = {}
        self.extent = np.zeros(len(self.tid))
        for i in range(len(self.tid)):                  # 전부 메모리에 올린다 (fork 한 작업자가 h5 를 다시 읽지 않게)
            p = self.points(i)
            self.extent[i] = float((p.max(0) - p.min(0)).max()) if len(p) else 0.0

    def points(self, i):
        if i not in self._pts:
            self._pts[i] = self.f[f'tracklets/{int(self.tid[i]):05d}/tracklet_geometry/points'][:].astype(np.float64)
        return self._pts[i]

    def active(self, frame, margin):
        return np.flatnonzero((self.first - margin <= frame) & (self.last + margin >= frame))


def pct(d, ps=(50, 90)):
    return {f'p{p}': (round(float(np.percentile(d, p)), 4) if len(d) else None) for p in ps}


def json_safe(x):
    """NaN·inf → None (null). JSON 표준(RFC 8259)에 NaN 토큰이 없다."""
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def write_csv(path, rows, fieldnames=None):
    """열 순서 = fieldnames(주면) 또는 처음 나온 순서. 행이 없으면 빈 파일 — 옛 결과가 남아 섞이지 않게 항상 쓴다."""
    with open(path, 'w', newline='') as fh:
        if rows:
            keys = list(fieldnames) if fieldnames else list(dict.fromkeys(k for r in rows for k in r))
            w = csv.DictWriter(fh, fieldnames=keys, restval=''); w.writeheader(); w.writerows(rows)


def load_vis(vis):
    """gt_visibility.py 출력(npz 경로) 또는 같은 키의 dict → {(ep, frame): px_crop}, 에피소드별 요약."""
    if vis is None:
        return None
    z = np.load(vis) if not isinstance(vis, dict) else vis
    ep, fr, pc = (np.asarray(z[k]) for k in ('ep_index', 'frame', 'px_crop'))
    return dict(ep=ep, frame=fr, px_crop=pc, pair={(int(e), int(f)): int(p) for e, f, p in zip(ep, fr, pc)})


def obs_columns(taus, T):
    """score_observations.csv 열 순서 — 옛 열(예전 dict 삽입 순서) 다음에 새 열."""
    cols = ['kf', 'frame', 'obs', 'tracklet_id', 'n_points', 'n_vox', 'top_oid', 'top_frac', 'second_frac', f'gt_counts@{T}']
    for t in taus:
        cols.append(f'match@{tau_cm(t)}')
        if tau_cm(t) == T:
            cols.append('dup_of')
    cols += ['ep_index', 'detect_credit', 'dist_med_cm', 'dist_mean_cm', 'dist_p90_cm', 'within_tau', 'voxel_exact',
             f'gt_vox@{T}', f'gt_iou@{T}', f'match_iou@{T}', f'match_P@{T}', f'match_R@{T}', 'pred_status', 'excluded_reason',
             f'void_frac@{T}', f'ignore_frac@{T}', 'dup_of_ep', 'tide_error']
    return cols


def kfgt_columns(taus):
    return (['kf', 'frame', 'ep_index', 'gt_object_id', 'is_human', 'px_label_5m', 'n_gt_vox', 'present', 'vis_px_crop', 'status',
             'matched_obs', 'matched_tracklet_id', 'match_iou', 'best_iou', 'best_obs', 'best_P', 'best_R', 'n_frags', 'split_cover',
             'merged_obs'] + [f'matched@{tau_cm(t)}' for t in taus])


# ------------------------------------------------------------------------------------------------------------------------
# keyframe 하나 채점 (작업자 프로세스에서도 돈다 — 전역 _CTX 는 fork 로 물려받는다)
# ------------------------------------------------------------------------------------------------------------------------
_CTX = {}


def _h5(ctx):
    """프로세스마다 frontend h5 를 새로 연다 (h5py 핸들은 fork 로 넘기지 않는다). 라벨 표면도 작업자에서 새로 만든다."""
    if ctx.get('_pid') != os.getpid():
        if ctx.get('_pid') is not None or ctx.get('_forked'):
            if ctx.get('surface_args'):
                poses = ctx['surface'].poses
                ctx['surface'] = GS.LabelSurface(*ctx['surface_args'])
                ctx['surface'].poses = poses
        ctx['_f'] = h5py.File(ctx['h5_path'], 'r'); ctx['_pid'] = os.getpid()
        ctx['_ref'] = {}
    return ctx['_f']


def _ref(ctx, ep):
    """에피소드 점군 → (KD-tree, 정렬된 셀 키). 작업자마다 캐시 (큰 물체 재구축 방지)."""
    cache = ctx['_ref']
    if ep not in cache:
        if len(cache) > 512:
            cache.clear()
        ref = ctx['gt'].points(ep)
        cache[ep] = (cKDTree(ref), np.sort(vkeys(ref, ctx['v'])))
    return cache[ep]


def _score_kf(k):
    ctx = _CTX
    f = _h5(ctx)
    gt, v, taus, tau, T = ctx['gt'], ctx['v'], ctx['taus'], ctx['tau'], ctx['T']
    thr, V = ctx['thr'], ctx['V']
    fi = int(ctx['kf_frame'][k]); o0, no = int(ctx['kf_obs0'][k]), int(ctx['kf_nobs'][k])
    surf = ctx['surface'].at(fi)
    eps_k = sorted(surf)                                      # GT 순서 = 에피소드 번호 오름차순 (score_mot 전처리와 같다)
    gvox = [surf[e].vox for e in eps_k]
    gpx = np.array([surf[e].px for e in eps_k], np.int64)
    present = gpx >= ctx['area_min_px']
    g_oid = np.array([int(gt.oid[e]) for e in eps_k], np.int64)
    o_tid, o_p0, o_pn = ctx['o_tid'], ctx['o_p0'], ctx['o_pn']
    pvox, n_pts, n_raw, n_far = [], [], 0, 0
    if no:
        lo, hi = int(o_p0[o0]), int(o_p0[o0 + no - 1] + o_pn[o0 + no - 1])
        block = f['points'][lo:hi].astype(np.float64) if hi > lo else np.zeros((0, 3))
    for o in range(o0, o0 + no):
        p = block[int(o_p0[o]) - lo:int(o_p0[o] + o_pn[o]) - lo]
        n_raw += len(p)
        keep = np.linalg.norm(p - ctx['cam_t'][k], axis=1) <= ctx['rng']
        n_far += int((~keep).sum())
        pvox.append(voxelize(p[keep], v)); n_pts.append(len(p))
    st = M.pair_stats(pvox, gvox, taus, tau)
    ti_main = st.main
    Np = len(pvox)
    # τ 마다 1:1 매칭 (행 GT j, 열 예측 i)
    assign = []                                               # τ 인덱스 → {예측 i: GT j}
    hung = None
    for ti in range(len(taus)):
        iou = st.iou(ti)
        r, c = M.match(iou, thr)
        a = {int(i): int(j) for j, i in zip(r, c)}
        if ti == ti_main:
            hung = (r, c)
        if ctx['dup_policy'] == 'count':                      # 옛 방식: IoU ≥ thr 인 중복도 매칭으로 친다 (최고 IoU GT)
            for i in range(Np):
                if i not in a and len(eps_k) and iou[:, i].max() >= thr - M.FEPS:
                    a[i] = int(np.argmax(iou[:, i]))
        assign.append(a)
    stat = M.gt_statuses(st, *hung, threshold=thr)
    iou_m, P_m, R_m = st.iou(ti_main), st.P(ti_main), st.R(ti_main)
    obs_rows, cov = [], defaultdict(list)
    for i in range(Np):
        o = o0 + i; n = int(st.n_pred[i])
        near = st.near[:, i]
        by_oid = Counter()
        for j in np.flatnonzero(near):
            by_oid[int(g_oid[j])] += int(near[j])
        ranked = sorted(by_oid.items(), key=lambda kv: -kv[1])[:2]
        nv = max(n, 1)
        row = dict(kf=k, frame=fi, obs=int(o), tracklet_id=int(o_tid[o]), n_points=int(n_pts[i]), n_vox=n,
                   top_oid=int(ranked[0][0]) if ranked else NONE, top_frac=round(ranked[0][1] / nv, 4) if ranked else 0.0,
                   second_frac=round(ranked[1][1] / nv, 4) if len(ranked) > 1 else 0.0)
        row[f'gt_counts@{T}'] = format_counts(by_oid)
        for ti, t in enumerate(taus):
            j = assign[ti].get(i)
            row[f'match@{tau_cm(t)}'] = int(g_oid[j]) if j is not None else NONE
        kind, dj = M.pred_error(st, i, *hung, valid=present) if i not in assign[ti_main] else ('', -1)   # [E2] TIDE (present = 무시 아닌 GT)
        dj = dj if kind == 'dupe' else -1
        row['dup_of'] = int(g_oid[dj]) if dj >= 0 else NONE
        row['dup_of_ep'] = int(eps_k[dj]) if dj >= 0 else ''
        row['tide_error'] = kind
        row[f'gt_vox@{T}'] = format_counts({eps_k[j]: int(near[j]) for j in np.flatnonzero(near)})
        row[f'gt_iou@{T}'] = M.format_values({eps_k[j]: iou_m[j, i] for j in np.flatnonzero(iou_m[:, i] > 0)}) if len(eps_k) else ''
        ign = int(st.void[i]) + int(near[~present].sum()) if len(eps_k) else int(st.void[i])
        row[f'void_frac@{T}'] = round(int(st.void[i]) / n, 4) if n else ''
        row[f'ignore_frac@{T}'] = round(ign / n, 4) if n else ''
        j = assign[ti_main].get(i)
        if j is None:
            status, reason = M.unmatched_class(n, int(st.void[i]), near, ~present)
            row.update(pred_status=status, excluded_reason=reason)
        else:
            e = int(eps_k[j]); ok = bool(present[j])
            rtree, rkeys = _ref(ctx, e)
            dd = rtree.query(pvox[i], k=1)[0]
            qk = vkeys(pvox[i], v)
            pos = np.clip(np.searchsorted(rkeys, qk), 0, len(rkeys) - 1)
            row.update(ep_index=e, detect_credit=int(ok), dist_med_cm=round(float(np.median(dd)) * 100, 2),
                       dist_mean_cm=round(float(np.mean(dd)) * 100, 2), dist_p90_cm=round(float(np.percentile(dd, 90)) * 100, 2),
                       within_tau=round(float(M.within(dd, tau).mean()), 4), voxel_exact=round(float((rkeys[pos] == qk).mean()), 4),
                       pred_status='tp' if ok else 'ignored_match', excluded_reason='')
            row[f'match_iou@{T}'] = round(float(iou_m[j, i]), 10)
            row[f'match_P@{T}'] = round(float(P_m[j, i]), 10)
            row[f'match_R@{T}'] = round(float(R_m[j, i]), 10)
            if ok:
                cov[e].append(pvox[i])
        obs_rows.append(row)
    kfgt_rows, ep_upd = [], []
    for j, e in enumerate(eps_k):
        s = stat[j]
        matched_i = [i for i, jj in assign[ti_main].items() if jj == j]
        m_i = s['matched_pred'] if s['matched_pred'] >= 0 else (matched_i[0] if matched_i else -1)
        if present[j]:
            status = 'tp' if m_i >= 0 else s['status']
        else:
            status = 'ignored_match' if m_i >= 0 else 'not_present'
        best = s['best_pred']
        row = dict(kf=k, frame=fi, ep_index=int(e), gt_object_id=int(g_oid[j]),
                   is_human=int(str(gt.color[e]).lstrip('#') in HUMAN_COLORS), px_label_5m=int(gpx[j]), n_gt_vox=int(st.n_gt[j]),
                   present=int(present[j]), vis_px_crop=(V['pair'].get((int(e), fi), 0) if V is not None else ''), status=status,
                   matched_obs=int(o0 + m_i) if m_i >= 0 else -1, matched_tracklet_id=int(o_tid[o0 + m_i]) if m_i >= 0 else -1,
                   match_iou=round(float(iou_m[j, m_i]), 10) if m_i >= 0 else '', best_iou=round(s['best_iou'], 10),
                   best_obs=int(o0 + best) if best >= 0 else -1, best_P=round(s['best_P'], 6), best_R=round(s['best_R'], 6),
                   n_frags=s['n_frags'], split_cover=round(s['split_cover'], 4),
                   merged_obs=int(o0 + s['merged_pred']) if s['merged_pred'] >= 0 else -1)
        for ti, t in enumerate(taus):
            row[f'matched@{tau_cm(t)}'] = int(any(jj == j for jj in assign[ti].values()))
        kfgt_rows.append(row)
        ep_upd.append(dict(ep=int(e), present=bool(present[j]), status=status, best_iou=s['best_iou'],
                           credit={ti: bool(present[j]) and any(jj == j for jj in assign[ti].values()) for ti in range(len(taus))},
                           ignored=(not present[j]) and m_i >= 0))
    return dict(k=k, obs=obs_rows, kfgt=kfgt_rows, ep=ep_upd, cov=dict(cov), n_raw=n_raw, n_far=n_far)


def score(run_dir, seq_dir, gt_path, tau=0.20, min_frac=0.5, absorb_frac=0.2, taus=(0.10, 0.20, 0.50),
          out_dir=None, max_range=None, gt_obj=None, vis=None, area_min_px=1600, dup_policy='hungarian', log=print,
          labels_dir=None, surface=None, workers=1):
    """labels_dir: gt_labels_2d.py 라벨 폴더 (keyframe 마다 <frame>.png + meta.json) — 3D 매칭의 GT 표면. 필수(surface 를 안 줄 때).
    surface:  at(frame) → {ep: gt_surface.Surface} · meta() 를 가진 객체 (테스트용 대역). 주면 labels_dir 대신 쓴다.
    min_frac: IoU_τ 임계 (이름은 옛 호환 — 예전엔 share 임계였다). 기본 0.5.
    vis:      gt_visibility.py npz. 채점 대상(eligible = crop px 최대 ≥ area_min_px)·시야 안 재현율과 라벨/가시성 present 비교에 쓴다.
    area_min_px: frontend 설계상 최소 마스크(sam.AREA_MIN 256 proto px × 2.5² = 원본 1600px) — present 와 eligible 에 같이.
    dup_policy: 'hungarian'(기본) = keyframe 안에서 GT 당 관측 1개만 매칭, 나머지는 중복(dup_of).
                'count' = 옛 방식, IoU ≥ 임계인 중복도 매칭으로 친다 (Khronos compensate_missegmentation 에 해당).
    workers:  keyframe 을 병렬로 채점할 프로세스 수 (fork). 1 이면 이 프로세스에서."""
    t0 = time.time()
    V = load_vis(vis)
    run_dir = Path(run_dir); out_dir = Path(out_dir or run_dir)
    gt = gt_obj or GT(gt_path)
    v = gt.voxel; margin = gt.gap
    rng = gt.max_range if max_range is None else max_range
    if surface is None:
        if labels_dir is None:
            raise ValueError('labels_dir 가 필요하다 — 3D 매칭 GT 는 keyframe GT 라벨(gt_labels_2d.py) 역투영 표면이다')
        surface = GS.LabelSurface(seq_dir, labels_dir, max_range=gt.max_range, voxel=v)
        surface_args = (seq_dir, labels_dir, gt.max_range, v)
    else:
        surface_args = None
    lmeta = surface.meta()

    seq = UH2Sequence(seq_dir)
    with h5py.File(run_dir / 'frontend_output.h5', 'r') as f:
        meta = json.loads(f.attrs['meta_json'])
        kf_frame = f['kf/frame_idx'][:]; kf_obs0 = f['kf/obs_start'][:]; kf_nobs = f['kf/n_obs'][:]
        o_tid = f['obs/tracklet_id'][:]; o_p0 = f['obs/points_start'][:]; o_pn = f['obs/points_num'][:]
    fr_lo, fr_hi = meta['frames']
    if hasattr(surface, 'prefetch_poses'):
        surface.prefetch_poses(kf_frame)
    cam_t, _ = seq.camera_poses(seq.stamps_ns[kf_frame], cam='left_cam') if len(kf_frame) else (np.zeros((0, 3)), None)

    taus = tuple(sorted(set(float(t) for t in list(taus) + [tau])))
    T = tau_cm(tau)
    counts_col, iou_col, vox_col = f'gt_counts@{T}', f'gt_iou@{T}', f'gt_vox@{T}'
    _CTX.clear()
    _CTX.update(gt=gt, v=v, taus=taus, tau=float(tau), T=T, thr=float(min_frac), V=V, surface=surface, area_min_px=area_min_px,
                dup_policy=dup_policy, rng=rng, cam_t=cam_t, kf_frame=kf_frame, kf_obs0=kf_obs0, kf_nobs=kf_nobs, o_tid=o_tid,
                o_p0=o_p0, o_pn=o_pn, h5_path=str(run_dir / 'frontend_output.h5'), surface_args=surface_args)
    n_kf = len(kf_frame)
    if workers > 1 and n_kf > 1:
        _CTX['_forked'] = True
        import multiprocessing as mp
        with mp.get_context('fork').Pool(workers) as pool:
            results = list(pool.imap(_score_kf, range(n_kf), chunksize=4))
    else:
        results = [_score_kf(k) for k in range(n_kf)]
    if _CTX.get('_f') is not None and _CTX.get('_pid') == os.getpid():
        _CTX['_f'].close()
    _CTX.clear()

    obs_rows, kfgt_rows = [], []
    ep_best = defaultdict(lambda: {t: 0 for t in taus})       # 에피소드 → tau별 탐지 인정 관측 수
    ep_nocredit = defaultdict(int)                            # 에피소드 → 제외 매칭(present 아닌 keyframe 의 매칭) 수 (주 τ)
    ep_vox = defaultdict(list)                                # 에피소드 → 탐지 인정된 관측 복셀들 (주 τ)
    ep_surf = defaultdict(int); ep_pres = defaultdict(int)
    ep_stat = defaultdict(list); ep_bestiou = defaultdict(float)
    label_present = set()
    n_pts_raw = n_pts_far = 0
    for r in results:
        obs_rows.extend(r['obs']); kfgt_rows.extend(r['kfgt'])
        n_pts_raw += r['n_raw']; n_pts_far += r['n_far']
        fi = int(kf_frame[r['k']])
        for u in r['ep']:
            e = u['ep']
            ep_surf[e] += 1
            if u['present']:
                ep_bestiou[e] = max(ep_bestiou[e], u['best_iou'])       # present keyframe 에서의 최고 IoU (제외 매칭은 안 셈)
                ep_pres[e] += 1; label_present.add((e, fi))
                if u['status'] != 'tp':
                    ep_stat[e].append(u['status'])
            for ti, t in enumerate(taus):
                ep_best[e][t] += int(u['credit'][ti])
            ep_nocredit[e] += int(u['ignored'])
        for e, vs in r['cov'].items():
            ep_vox[e].extend(vs)

    # ---------------- 에피소드 (놓친 것) ----------------
    in_run = (gt.last >= fr_lo) & (gt.first < fr_hi)       # 부분 실행이면 그 구간에 걸친 에피소드만
    kf_set = set(int(x) for x in kf_frame)
    kf_frames_sorted = np.sort(kf_frame)
    vis_ep = {}
    if V is not None:                                      # 에피소드별: crop 가시 프레임 · 최대 가시 픽셀 · 보인 동안의 kf 수
        for e, fr, pc in zip(V['ep'], V['frame'], V['px_crop']):
            if pc <= 0 or not (fr_lo <= int(fr) < fr_hi):  # 실행 구간 밖 프레임의 가시성은 세지 않는다 (감사 A-8)
                continue
            s = vis_ep.setdefault(int(e), dict(frames=0, max_px=0, kf_visible=0, frames_ge_min=0))
            s['frames'] += 1; s['max_px'] = max(s['max_px'], int(pc))
            s['kf_visible'] += int(int(fr) in kf_set); s['frames_ge_min'] += int(pc >= area_min_px)
    ep_rows = []
    for i in np.flatnonzero(in_run):
        i = int(i)
        lo, hi = gt.first[i], gt.last[i]
        nkf = int(np.searchsorted(kf_frames_sorted, hi, 'right') - np.searchsorted(kf_frames_sorted, lo, 'left'))
        ve = vis_ep.get(i, dict(frames=0, max_px=0, kf_visible=0, frames_ge_min=0))
        det = ep_best[i][tau] > 0
        reason = '' if det else M.track_reason(ep_pres[i], ep_stat[i])
        cov = None
        if det:
            pv = np.concatenate(ep_vox[i])
            cov = round(float(M.within(cKDTree(pv).query(gt.points(i), k=1)[0], tau).mean()), 4)
        sc = Counter(ep_stat[i])
        is_human = int(str(getattr(gt, 'color', [''] * len(gt.tid))[i]).lstrip('#') in HUMAN_COLORS)
        r = dict(ep_index=i, gt_tracklet_id=int(gt.tid[i]), gt_object_id=int(gt.oid[i]), is_human=is_human,
                 first_frame=int(gt.first[i]), last_frame=int(gt.last[i]), n_frames_observed=int(gt.nfo[i]),
                 n_points=int(gt.npts[i]), extent_m=round(float(gt.extent[i]), 3), dist_min_m=round(float(gt.dist_min[i]), 3),
                 n_kf_in_window=nkf, detected=int(det), miss_reason=reason,
                 n_matched_obs=int(ep_best[i][tau]), n_absorbed_obs=int(sc.get('merged', 0)), coverage_at_tau=cov,
                 n_matched_obs_no_credit=int(ep_nocredit[i]))
        if V is not None:
            r.update(vis_frames_crop=ve['frames'], max_px_crop=ve['max_px'], n_kf_while_visible=ve['kf_visible'],
                     frames_px_ge_min=ve['frames_ge_min'])
        for t in taus:
            r[f'detected@{tau_cm(t)}'] = int(ep_best[i][t] > 0)
        r.update(n_kf_with_surface=int(ep_surf[i]), n_kf_present=int(ep_pres[i]),
                 **{f'n_kf_{s_}': int(sc.get(s_, 0)) for s_ in M.STATUS_ORDER},
                 best_iou=round(float(ep_bestiou[i]), 6) if ep_pres[i] else '')
        ep_rows.append(r)

    # keyframe × GT (라벨 present 기준 — score_mot 의 GT 검출과 같은 집합, 사람 포함)
    def is_hum(e):
        return str(getattr(gt, 'color', [''] * len(gt.tid))[e]).lstrip('#') in HUMAN_COLORS

    pres_rows = [r for r in kfgt_rows if r['present'] and in_run[r['ep_index']]]
    hit = np.array([r['status'] == 'tp' for r in pres_rows], bool)
    pxs = np.array([r['px_label_5m'] for r in pres_rows], np.int64)
    hum = np.array([is_hum(r['ep_index']) for r in pres_rows], bool)

    def rate_of(sel):
        sel = np.asarray(sel, bool)
        return round(float(hit[sel].mean()), 4) if sel.any() else None

    kf_level = dict(n_pairs=len(pres_rows), recall=rate_of(np.ones(len(pres_rows), bool)),
                    by_visible_px=[dict(bin=lb, n=int(((pxs >= lo_) & (pxs < hi_)).sum()), recall=rate_of((pxs >= lo_) & (pxs < hi_)))
                                   for lo_, hi_, lb in ((area_min_px, 6400, '1600-6399px'), (6400, 25600, '6400-25599px'),
                                                        (25600, 10 ** 9, '25600px+'))],
                    by_kind={kd: dict(n=int(s_.sum()), recall=rate_of(s_)) for kd, s_ in (('static', ~hum), ('human', hum))})
    presence_vs_vis = None
    if V is not None:
        vis_present = {(int(e), int(fr)) for (e, fr), pc in V['pair'].items()
                       if fr in kf_set and pc >= area_min_px and e < len(in_run) and in_run[e]}
        lab_present = {p for p in label_present if in_run[p[0]]}
        both = len(vis_present & lab_present); uni = len(vis_present | lab_present)
        presence_vs_vis = dict(n_label_present=len(lab_present), n_vis_present=len(vis_present), n_both=both,
                               n_label_only=len(lab_present - vis_present), n_vis_only=len(vis_present - lab_present),
                               disagreement_rate=round((uni - both) / uni, 4) if uni else None)
    kst = Counter(r['status'] for r in kfgt_rows if in_run[r['ep_index']])
    keyframe_gt = dict(
        n_pairs_with_surface=int(sum(kst.values())), n_present=len(pres_rows), n_tp=int(hit.sum()),
        recall=kf_level['recall'], status=dict(kst), by_kind=kf_level['by_kind'],
        miss_status_present=dict(Counter(r['status'] for r in pres_rows if r['status'] != 'tp')),
        presence_vs_vis=presence_vs_vis)

    # ---------------- 요약 ----------------
    E = len(ep_rows)
    det = np.array([r['detected'] for r in ep_rows], bool)

    def rate(sel):
        """선택된 에피소드의 재현율. 선택이 비면 null."""
        sel = np.asarray(sel, bool)
        return round(float(det[sel].mean()), 4) if sel.any() else None

    reasons = Counter(r['miss_reason'] for r in ep_rows if not r['detected'])
    objs = defaultdict(bool)
    for r in ep_rows:
        objs[r['gt_object_id']] |= bool(r['detected'])
    O = len(obs_rows)
    m_obs = [r for r in obs_rows if r[f'match@{T}'] != NONE]
    dmed = np.array([r['dist_med_cm'] for r in m_obs]); dmean = np.array([r['dist_mean_cm'] for r in m_obs])
    wtau = np.array([r['within_tau'] for r in m_obs]); vex = np.array([r['voxel_exact'] for r in m_obs])
    covs = np.array([r['coverage_at_tau'] for r in ep_rows if r['detected']])
    pst = Counter(r['pred_status'] for r in obs_rows)
    n_dup = int(sum(r['dup_of'] != NONE for r in obs_rows))

    def bins(key, edges, labels):
        out = []
        vals = np.array([r[key] for r in ep_rows], np.float64)
        for (lo, hi), lb in zip(zip(edges[:-1], edges[1:]), labels):
            s = (vals >= lo) & (vals < hi)
            out.append(dict(bin=lb, n=int(s.sum()), recall=rate(s)))
        return out

    visibility = None
    if V is not None:
        mpx = np.array([r['max_px_crop'] for r in ep_rows], np.int64)
        hum_ep = np.array([r['is_human'] for r in ep_rows], bool)
        detectable = mpx >= area_min_px
        visibility = dict(
            area_min_px=area_min_px,
            n_in_view=int((mpx > 0).sum()),
            recall_in_view=rate(mpx > 0),
            n_detectable=int(detectable.sum()),
            recall_detectable=rate(detectable),
            miss_reasons_detectable=dict(Counter(r['miss_reason'] for r, s_ in zip(ep_rows, detectable) if s_ and not r['detected'])),
            by_kind={kind: dict(
                n_detectable=int(sel.sum()), recall_detectable=rate(sel),
                miss_reasons_detectable=dict(Counter(r['miss_reason'] for r, s_ in zip(ep_rows, sel) if s_ and not r['detected'])))
                for kind, sel in (('static', detectable & ~hum_ep), ('human', detectable & hum_ep))},
            recall_by_max_px_crop=bins('max_px_crop', [0, 1, 400, 1600, 6400, 25600, 10 ** 12],
                                       ['0 (시야 밖)', '1-399', '400-1599', '1600-6399', '6400-25599', '25600+']),
            keyframe_level=kf_level)

    summary = dict(
        sequence=meta['sequence'], frames=meta['frames'], run=str(run_dir), gt=str(gt_path),
        params=dict(tau_m=tau, min_frac=min_frac, absorb_frac=absorb_frac, voxel_m=v, max_range_m=rng,
                    matching=f'hungarian 1:1 per keyframe (cost = tolerant IoU_tau vs keyframe GT label surface, candidates '
                             f'IoU >= {min_frac}, no temporal term)', dup_policy=dup_policy,
                    window_margin_frames=margin, sensitivity_taus_m=list(taus),
                    dist_eps_m=EPS, counts_column=counts_col,
                    detect_window=f'keyframes where the GT track is present (label px within {gt.max_range:g}m >= {area_min_px})',
                    label_window='not used for matching since 26/09/17 (GT = per-keyframe 2D label surface); '
                                 '[first_frame - window_margin_frames, last_frame + window_margin_frames] is only the old voxel-label rule',
                    # --- 26/09/17 결정 ④ 추가 ---
                    similarity=SIMILARITY,
                    iou_threshold=min_frac, loc_min_iou=M.TR.LOC_MIN_IOU, iou_column=iou_col, vox_column=vox_col,
                    kf_gt_file='score_kf_gt.csv',
                    gt_surface=f'2D GT label pixels (ep+1) back-projected with keyframe depth, K, GT pose; range <= {gt.max_range:g}m; '
                               f'{v:g}m voxels (same as score_geometry)',
                    labels_dir=lmeta.get('labels_dir'), labels_version=lmeta.get('version'), labels_meta_sha1=lmeta.get('meta_sha1'),
                    presence=f'label px within {gt.max_range:g}m >= {area_min_px} at that keyframe; match to non-present GT = ignored match',
                    eligible=f'gt_vis max px_crop >= {area_min_px} (unchanged)',
                    fp_rule='unmatched pred excluded if (void + voxels nearest to non-present GT) / |P| > 0.5 (panopticapi), else FP',
                    dup_rule=f'TIDE Dupe (tidecv quantify.py:250-255): unmatched pred with IoU_tau >= {min_frac} to a present GT '
                             f'matched by another pred in the keyframe; Loc ([{M.TR.LOC_MIN_IOU}, {min_frac}], quantify.py:236-240) '
                             'is tested first',
                    tide_error=f'unmatched preds: dupe | loc (max IoU_tau in [{M.TR.LOC_MIN_IOU}, {min_frac}]) | bkg (< {M.TR.LOC_MIN_IOU}) | '
                               'other, over present GTs (quantify.py:228-265); independent of fp/excluded',
                    merge_split_coverage='exclusive (nearest-GT ownership): GT voxel y is covered by pred i iff among its nearest pred '
                                         'voxels (d < tau, ties +-1e-9 m) one belongs to i and that voxel\'s nearest GT is this GT; '
                                         'split fragments use Px = pred voxels owned by the GT / |pred|; matching IoU_tau is non-exclusive',
                    miss_reasons=f'per (keyframe, present GT): merged (a pred with Rx>=0.5 on >=2 GTs covers it with Rx>=0.5) > split '
                                 f'(>=2 preds with Px>=0.5, exclusive union cover>=0.5) > low_iou (best IoU_tau in [{M.TR.LOC_MIN_IOU}, {min_frac}], '
                                 'tide_rules.is_loc) > missed; track: no_kf if never present at a keyframe, else most frequent status',
                    reason_tie_order=list(M.STATUS_ORDER), reason_mapping=dict(M.OLD_REASON_MAP),
                    detect_credit='1 = matched GT is present at that keyframe (credited), 0 = ignored match',
                    accuracy_tau_rule='d < tau (strict); reference = matched episode cloud',
                    absorb_frac_use='only multi_object_obs_rate (second GT fraction)',
                    n_absorbed_obs='deprecated: = number of keyframes with status merged'),
        frontend=dict(n_kf=int(len(kf_frame)), n_obs=O, n_tracklet_ids=int(len(np.unique(o_tid))) if O else 0,
                      points_dropped_beyond_range=round(n_pts_far / max(n_pts_raw, 1), 4)),
        missed=dict(
            n_gt_episodes=E, detected=int(det.sum()), recall=rate(np.ones(E, bool)),
            missed=int((~det).sum()), miss_reasons=dict(reasons),
            n_gt_objects=len(objs), objects_detected=int(sum(objs.values())),
            object_recall=round(sum(objs.values()) / len(objs), 4) if objs else None,
            recall_by_frames_observed=bins('n_frames_observed', [0, 10, 30, 100, 10 ** 9], ['<10', '10-29', '30-99', '100+']),
            recall_by_n_points=bins('n_points', [0, 100, 1000, 10000, 10 ** 12], ['<100', '100-999', '1k-9.9k', '10k+']),
            recall_by_dist_min=bins('dist_min_m', [0, 1, 2, 3, 99], ['<1m', '1-2m', '2-3m', '3m+']),
            recall_excluding_extent_over_10m=rate([r['extent_m'] <= 10 for r in ep_rows])),
        visibility=visibility,
        accuracy=dict(
            n_obs_matched=len(m_obs), obs_match_rate=round(len(m_obs) / O, 4) if O else None,
            n_obs_duplicate=n_dup, duplicate_rate=round(n_dup / O, 4) if O else None,
            n_obs_matched_no_credit=int(sum(not r.get('detect_credit') for r in m_obs)),
            dist_error_cm=dict(per_obs_median=pct(dmed, (25, 50, 75, 90)), per_obs_mean=pct(dmean, (50, 90))),
            obs_median_error_within_tau=round(float((dmed < T).mean()), 4) if len(dmed) else None,
            within_tau_fraction_mean=round(float(wtau.mean()), 4) if len(wtau) else None,
            voxel_exact_2cm_mean=round(float(vex.mean()), 4) if len(vex) else None,
            episode_coverage_at_tau=pct(covs, (25, 50, 75)),
            multi_object_obs_rate=round(float(np.mean([r['second_frac'] >= absorb_frac for r in m_obs])), 4) if m_obs else None,
            # --- 26/09/17 결정 ④ 추가 ---
            n_obs_tp=int(pst.get('tp', 0)), n_obs_ignored_match=int(pst.get('ignored_match', 0)),
            n_obs_fp=int(pst.get('fp', 0)), n_obs_excluded=int(pst.get('excluded', 0)),
            excluded_reasons=dict(Counter(r['excluded_reason'] for r in obs_rows if r['pred_status'] == 'excluded')),
            n_obs_duplicate_fp=int(sum(r['dup_of'] != NONE and r['pred_status'] == 'fp' for r in obs_rows)),
            # --- 수정 ⑤ E2 추가: 매칭 안 된 예측의 TIDE 오류 분포 (FP / 제외와 교차) ---
            unmatched_tide_errors=dict(Counter(r['tide_error'] for r in obs_rows if r['tide_error'])),
            unmatched_tide_errors_by_status={st_: dict(Counter(r['tide_error'] for r in obs_rows if r['pred_status'] == st_))
                                             for st_ in ('fp', 'excluded')}),
        keyframe_gt=keyframe_gt,
        sensitivity=[dict(tau_cm=tau_cm(t),
                          recall=round(float(np.mean([r[f'detected@{tau_cm(t)}'] for r in ep_rows])), 4) if E else None,
                          obs_match_rate=round(float(np.mean([r[f'match@{tau_cm(t)}'] != NONE for r in obs_rows])), 4) if O else None)
                     for t in taus],
        seconds=round(time.time() - t0, 1))
    summary = json_safe(summary)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'score.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    write_csv(out_dir / 'score_episodes.csv', ep_rows)
    write_csv(out_dir / 'score_observations.csv', obs_rows, obs_columns(taus, T))
    write_csv(out_dir / 'score_kf_gt.csv', kfgt_rows, kfgt_columns(taus))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--seq', required=True); ap.add_argument('--gt', required=True)
    ap.add_argument('--labels', required=True, help='gt_labels_2d.py 라벨 폴더 — 3D 매칭 GT 표면')
    ap.add_argument('--tau', type=float, default=0.20)
    ap.add_argument('--min-frac', '--iou-threshold', dest='min_frac', type=float, default=0.5, help='IoU_τ 임계')
    ap.add_argument('--absorb-frac', type=float, default=0.2); ap.add_argument('--out', default=None)
    ap.add_argument('--vis', default=None, help='gt_visibility.py npz — 주면 시야·탐지가능 기준 재현율 추가')
    ap.add_argument('--area-min-px', type=int, default=1600)
    ap.add_argument('--dup-policy', choices=['hungarian', 'count'], default='hungarian')
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    s = score(a.run, a.seq, a.gt, a.tau, a.min_frac, a.absorb_frac, out_dir=a.out, vis=a.vis, area_min_px=a.area_min_px,
              dup_policy=a.dup_policy, labels_dir=a.labels, workers=a.workers)
    print(json.dumps(s, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
