#!/usr/bin/env python3
"""score_frontend.py 자체 검증. 채점기가 틀리면 frontend 평가 전체가 틀리므로, 답을 손으로 아는 입력으로 확인한다.

3D 매칭 기준 (26/09/17 사용자 결정 ④): keyframe 마다 그 keyframe 에 보인 GT 표면(2D GT 라벨 역투영 · 5m · 2cm 복셀)과의
허용 거리 IoU_τ = P·R/(P+R−P·R) ≥ 0.5 헝가리안. 합성 사례는 GT 표면을 FakeSurface(gt_surface.LabelSurface 대역)로 준다:
프레임 f 의 표면 = f 에 활성인 에피소드 점군, 라벨 px = 5000 (override 로 바꿈).
τ = 20cm = 2cm 격자에서 '셀 거리 < 10'. 같은 열/행 방향 판정은 '9셀 이하'.

A. 합성 장면 (옛 A 와 같은 입력, 기대값은 IoU 기준으로 다시 손계산)
   ep0 물체1 평면 A(50×50), 0~50       kf20 에서 법선 +16cm 로 밀어 발행      → P=R=1 → IoU 1 · tau10 은 IoU 0
   ep1 물체2 평면 B(30×30), 10~60      kf20 에서 B 900 + A 앞 600 셀           → IoU(B) = 0.6/(1.6−0.6) = 0.6 → 매칭
   ep2 물체3, 100~120                   그 기간 keyframe 없음                   → no_kf
   ep3 물체1 A, 200~250                 kf220 에서 A 2500 + 먼 배경 3025        → IoU = 2500/5525 < 0.5 → low_iou (옛 absorbed)
   ep4 물체4 D, 300~320                 kf310 에서 100m 밖                      → IoU 0 → missed (옛 not_detected)
   A2 가시성 추가 → 채점 대상(eligible) · keyframe 단위 재현율(라벨 present 기준) · 라벨/가시성 present 불일치
A3 중복 · A4 IoU 헝가리안(시간 정보 없음) · A5 strict τ 경계(float32) · A6 τmax 상한 · A7 부분 실행 · A8 빈 입력
A9 같은 물체 두 에피소드(라벨이 에피소드 단위라 원인이 섞이지 않음) · A10 present 기준 탐지 인정(detect_credit)
A11 관측별 열(gt_counts · gt_vox · gt_iou)
F1 조각 사례 (감사: apartment kf 29 / ep 78 — share 최대가 작은 순수 조각을 골랐다)
F2 제외 매칭(ignored match) · F3 present 1600px 경계 · F4 FP/제외 경계(정확히 0.5 · 초과) · F5 미검출 원인(merged·split·
   low_iou·missed) · F6 트랙 원인(no_kf · 최빈값) · F7 score_kf_gt.csv
F8 [E1] 얇은 문틀 GT 가 매칭된 벽 예측 20cm 안에 있어도 merged 가 아니다 (배타적 덮음) · [E2] 중복 = TIDE Dupe · tide_error 열
R. 실제 apartment_s1_00h keyframe 6개 (최신 라벨이 있을 때만)
   R1 GT 라벨 픽셀을 frontend 격자 밀도로 뽑은 '완벽 예측' → present GT 전부 매칭, IoU_τ ≈ 1 · params 에 라벨 version·sha1
   R2 +100m 이동 → 매칭 0 · present GT 전부 missed
   R3 present GT 3개 중 1개꼴로 예측을 뺌 → 검출 재현율 = 남긴 수 / present 수 정확히
실제 데이터 없이 합성만: python test_score.py --synthetic-only
"""
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # 채점 코드는 한 칸 위 (frontend_benchmark/)
import score_frontend as S  # noqa: E402
from paths import WS, MODELS, REAL_RUNS as RUNS  # noqa: E402,F401  (실제 데이터 — FB_RUNS 와 무관, 수정 ⑤ E4)
SEQ = WS / 'datasets/unpacked/uHumans2_apartment_s1_00h'
GTH5 = WS / 'src/meridian/meridian_benchmark/tracklets/uHumans2_apartment_s1_00h.h5'
V = 0.02
FAILS = []


def check(name, got, want, tol=1e-6):
    ok = (got == want) if not isinstance(want, float) else (got is not None and abs(float(got) - want) <= tol)
    print(f'  {"OK " if ok else "FAIL"} {name}: got={got} want={want}')
    if not ok:
        FAILS.append(name)


def write_run(path, kfs, n_frames):
    """kfs = [(frame, [(tracklet_id, points[M,3])...])] → run_frontend.py 와 같은 h5 레이아웃."""
    path.mkdir(parents=True, exist_ok=True)
    obs_tid, obs_pn, pts, kf_frame, kf_nobs = [], [], [], [], []
    for fr, obs in kfs:
        kf_frame.append(fr); kf_nobs.append(len(obs))
        for tid, p in obs:
            obs_tid.append(tid); obs_pn.append(len(p)); pts.append(np.asarray(p, np.float32))
    with h5py.File(path / 'frontend_output.h5', 'w') as f:
        f['kf/frame_idx'] = np.array(kf_frame, np.int64)
        f['kf/n_obs'] = np.array(kf_nobs, np.int64)
        f['kf/obs_start'] = np.concatenate(([0], np.cumsum(kf_nobs)[:-1])).astype(np.int64)
        f['obs/tracklet_id'] = np.array(obs_tid, np.int64)
        f['obs/points_num'] = np.array(obs_pn, np.int64)
        f['obs/points_start'] = np.concatenate(([0], np.cumsum(obs_pn)[:-1])).astype(np.int64)
        f['points'] = np.concatenate(pts) if pts else np.zeros((0, 3), np.float32)
        f.attrs['meta_json'] = json.dumps(dict(sequence='test', frames=[0, n_frames]))


def plane(x0, y0, z, nx, ny):
    """z 고정 평면의 2cm 셀 중심들."""
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny))
    return np.stack([(np.floor(x0 / V) + ix.ravel() + 0.5) * V, (np.floor(y0 / V) + iy.ravel() + 0.5) * V,
                     np.full(ix.size, (np.floor(z / V) + 0.5) * V)], 1)


def cells(i0, j0, k0, nx, ny):
    """정수 셀 번호로 만든 평면 (x 셀 i0.. · y 셀 j0.. · z 셀 k0) — 실수 나눗셈 반올림 걱정 없음."""
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny))
    return np.stack([(i0 + ix.ravel() + 0.5) * V, (j0 + iy.ravel() + 0.5) * V, np.full(ix.size, (k0 + 0.5) * V)], 1)


class FakeGT:
    def __init__(self, eps):
        self.voxel, self.max_range, self.gap = V, 5.0, 5
        self._p = [e['pts'] for e in eps]
        self.tid = np.arange(1, len(eps) + 1); self.oid = np.array([e['oid'] for e in eps])
        self.first = np.array([e['first'] for e in eps]); self.last = np.array([e['last'] for e in eps])
        self.nfo = self.last - self.first + 1; self.npts = np.array([len(p) for p in self._p])
        self.dist_min = np.ones(len(eps)); self.flushed = np.zeros(len(eps), int)
        self.extent = np.array([float((p.max(0) - p.min(0)).max()) for p in self._p])
        self.color = np.array([e.get('color', 'aaaaaa') for e in eps])

    def points(self, i):
        return self._p[i]

    def active(self, frame, margin):
        return np.flatnonzero((self.first - margin <= frame) & (self.last + margin >= frame))


class FakeSurface:
    """gt_surface.LabelSurface 대역 (at(frame) → {ep: Surface(vox, px)} · meta()).
    기본: 프레임 f 에 활성(first ≤ f ≤ last)인 에피소드마다 표면 = 에피소드 점군 전부, 라벨 px = px.
    pairs = {(frame, ep): px} 를 주면 그 쌍에만 표면이 있다 (옛 테스트의 가시성 입력을 그대로 옮길 때).
    override = {(frame, ep): None(표면 없음) | px | (points, px)}."""

    def __init__(self, gt, px=5000, pairs=None, override=None):
        self.gt, self.px, self.pairs, self.override = gt, px, pairs, dict(override or {})

    def at(self, frame):
        import gt_surface as GS
        gt = self.gt
        if self.pairs is not None:
            raw = {int(e): (gt.points(int(e)), int(p)) for (f, e), p in self.pairs.items() if int(f) == int(frame)}
        else:
            raw = {int(i): (gt.points(int(i)), self.px) for i in np.flatnonzero((gt.first <= frame) & (gt.last >= frame))}
        for (f, e), val in self.override.items():
            if int(f) != int(frame):
                continue
            if val is None:
                raw.pop(e, None)
            elif isinstance(val, tuple):
                raw[e] = val
            else:
                raw[e] = (gt.points(e), int(val))
        return {e: GS.Surface(GS.voxelize(p, V), px) for e, (p, px) in raw.items() if len(p) and px > 0}

    def meta(self):
        return dict(labels_dir='FakeSurface', version=None, meta_sha1=None)


def read_csv(path):
    """없거나 빈 CSV → [] (에피소드·관측 0개일 때 채점기는 빈 파일을 쓴다)."""
    path = Path(path)
    return list(csv.DictReader(open(path))) if path.exists() and path.stat().st_size else []


def run_case(tmp, name, gt, kfs, n_frames=400, vis=None, surface=None, **kw):
    write_run(tmp / name, kfs, n_frames)
    s = S.score(tmp / name, SEQ, None, max_range=1e9, gt_obj=gt, out_dir=tmp / name, vis=vis, log=lambda *a: None,
                surface=surface if surface is not None else FakeSurface(gt), **kw)
    return s, read_csv(tmp / name / 'score_observations.csv'), read_csv(tmp / name / 'score_episodes.csv')


def strict_json(path):
    """RFC 8259 JSON 인지 (NaN·Infinity 토큰이면 예외)."""
    def bad(c):
        raise ValueError(f'비표준 JSON 토큰 {c}')
    return json.loads(Path(path).read_text(), parse_constant=bad)


def guarded(fn):
    """예외도 실패로 센다 — 테스트 하나가 죽어도 나머지는 계속 돈다."""
    def wrap(*a):
        try:
            fn(*a)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f'  FAIL {fn.__name__}: 예외 {type(e).__name__}: {e}')
            FAILS.append(f'{fn.__name__} 예외')
    return wrap


def ious(row, col='gt_iou@20'):
    import match3d as M
    return M.parse_values(row.get(col, ''))


@guarded
def test_synthetic(tmp):
    print('A. 합성 장면')
    A = plane(0, 0, 0.0, 50, 50)          # 1m x 1m, 2500 셀
    B = plane(3, 0, 0.0, 30, 30)          # 900 셀
    C = plane(0, 3, 0.0, 10, 10)
    D = plane(6, 6, 0.0, 10, 10)
    gt = FakeGT([dict(oid=1, first=0, last=50, pts=A), dict(oid=2, first=10, last=60, pts=B),
                 dict(oid=3, first=100, last=120, pts=C, color='23d5ea'), dict(oid=1, first=200, last=250, pts=A),
                 dict(oid=4, first=300, last=320, pts=D)])
    obs1 = A + [0, 0, 0.16]                              # 8셀 위 → 셀 중심 정확히 0.16m
    a_part = A[:600]                                     # 12행
    obs2 = np.concatenate([B, a_part])
    bg = plane(0, 0, 3.0, 55, 55)[:(2500 * 55) // 45]    # A 2500 + 먼 배경 (55×55 = 3025 셀뿐이라 슬라이스는 3025개)
    obs3 = np.concatenate([A, bg])
    obs4 = D + [100, 0, 0]
    kfs = [(20, [(11, obs1), (12, obs2)]), (220, [(13, obs3)]), (310, [(14, obs4)])]
    s, obs, eps = run_case(tmp, 'syn', gt, kfs)
    ep = {int(r['ep_index']): r for r in eps}
    ob = {int(r['tracklet_id']): r for r in obs}
    check('ep0 탐지', ep[0]['detected'], '1'); check('ep1 탐지', ep[1]['detected'], '1')
    check('ep2 원인 no_kf', ep[2]['miss_reason'], 'no_kf')
    # 옛 기대 'absorbed' (share 45% ≥ 20%) → IoU = 2500/5525 = 0.452 ∈ [0.25, 0.5) → low_iou (옛 값 absorbed·not_detected 폐지)
    check('ep3 원인 low_iou (옛 absorbed)', ep[3]['miss_reason'], 'low_iou')
    check('ep4 원인 missed (옛 not_detected)', ep[4]['miss_reason'], 'missed')
    check('재현율', s['missed']['recall'], 0.4)
    check('물체 재현율 (물체1·2 탐지 / 4개)', s['missed']['object_recall'], 0.5)
    check('obs1 IoU@20 = 1 (16cm < 20cm)', float(ob[11]['match_iou@20']), 1.0)
    check('obs1 거리 중앙값 cm', float(ob[11]['dist_med_cm']), 16.0, 0.01)
    check('obs1 복셀일치', float(ob[11]['voxel_exact']), 0.0)
    check('obs2 매칭 물체', ob[12]['match@20'], '2')
    check('obs2 IoU(B) = 0.6', float(ob[12]['match_iou@20']), 0.6, 1e-9)
    # IoU(A) = P 0.4 · R (A 12행 → 0..20행 = 1050/2500 = 0.42) → 0.168/0.652
    check('obs2 gt_iou@20 의 ep0 값 = 0.168/0.652', ious(ob[12]).get(0), 0.168 / 0.652, 1e-9)
    check('obs2 20cm 이내 비율 (ep1 점군 기준)', float(ob[12]['within_tau']), 0.6, 1e-4)
    check('obs2 복셀일치', float(ob[12]['voxel_exact']), 0.6, 1e-4)
    check('obs2 두번째 물체 비율', float(ob[12]['second_frac']), 0.4, 1e-4)
    check('obs3 매칭 없음', ob[13]['match@20'], '0')
    check('obs3 IoU = P = 2500/5525', ious(ob[13]).get(3), 2500 / 5525, 1e-9)
    check('obs3 제외 (void 3025/5525 > 0.5)', (ob[13]['pred_status'], ob[13]['excluded_reason']), ('excluded', 'void'))
    check('obs1·obs2 상태 tp', (ob[11]['pred_status'], ob[12]['pred_status']), ('tp', 'tp'))
    check('ep0 커버리지', float(ep[0]['coverage_at_tau']), 1.0)
    sens = {r['tau_cm']: r['recall'] for r in s['sensitivity']}
    check('tau10 재현율 (obs1 16cm 라 탈락)', sens[10], 0.2); check('tau50 재현율', sens[50], 0.4)
    check('다물체 관측 비율 (매칭 2개 중 obs2)', s['accuracy']['multi_object_obs_rate'], 0.5)
    check('매칭 기준 문자열에 IoU', 'IoU' in s['params']['matching'], True)

    print('A2. 합성 장면 + 가시성')
    # ep0 kf20 5000px · ep1 kf20 3000px · ep2 f110(kf 아님) 4000px · ep3 kf220 2000px · ep4 kf310 100px(설계 최소 미만)
    vis = dict(ep_index=np.array([0, 1, 2, 3, 4]), frame=np.array([20, 20, 110, 220, 310]),
               px_crop=np.array([5000, 3000, 4000, 2000, 100]))
    s, _, eps = run_case(tmp, 'syn_vis', gt, kfs, vis=vis)
    vv = s['visibility']
    check('시야 안 에피소드', vv['n_in_view'], 5); check('시야 안 재현율', vv['recall_in_view'], 0.4)
    check('탐지가능(≥1600px) 에피소드', vv['n_detectable'], 4); check('탐지가능 재현율', vv['recall_detectable'], 0.5)
    check('탐지가능 놓침 원인', vv['miss_reasons_detectable'], {'no_kf': 1, 'low_iou': 1})
    # keyframe 단위 = 라벨 present(≥1600px) 쌍: ep0·ep1@20 · ep3@220 · ep4@310 (라벨 5000px — 가시성 100px 과 무관)
    check('keyframe 단위 쌍 수 (라벨 present)', vv['keyframe_level']['n_pairs'], 4)
    check('keyframe 단위 재현율 2/4', vv['keyframe_level']['recall'], 0.5)
    ep = {int(r['ep_index']): r for r in eps}
    check('ep4 원인 missed (라벨로 present 인데 못 잡음)', ep[4]['miss_reason'], 'missed')
    bk = vv['by_kind']   # ep2 = 사람색 · 탐지가능 정적 = ep0·ep1·ep3
    check('정적 탐지가능 재현율 (2/3)', bk['static']['recall_detectable'], 0.6667)
    check('사람 탐지가능 (ep2, no_kf)', (bk['human']['n_detectable'], bk['human']['recall_detectable'],
                                   bk['human']['miss_reasons_detectable']), (1, 0.0, {'no_kf': 1}))
    pv = s['keyframe_gt']['presence_vs_vis']
    check('present 불일치: 둘 다 3 · 라벨만 1 (ep4@310) · 가시성만 0',
          (pv['n_both'], pv['n_label_only'], pv['n_vis_only']), (3, 1, 0))
    check('불일치율 1/4', pv['disagreement_rate'], 0.25)


@guarded
def test_duplicates(tmp):
    print('A3. 같은 물체를 한 keyframe 에 두 번 보냄 (TIDE Dupe)')
    A = plane(0, 0, 0.0, 50, 50)                         # 2500 셀
    far = plane(40, 40, 0.0, 20, 15)                     # 300 셀, GT 없음
    gt = FakeGT([dict(oid=1, first=0, last=50, pts=A)])
    # 옛 입력(A 앞 1200 + far 300)은 IoU 0.43 이라 'count' 방식에서도 매칭 후보가 아니다 → IoU ≥ 0.5 인 중복으로 바꿨다.
    # obs22 = A 앞 40행 + far: P = 2000/2300 · R = 0..48행 = 0.98 → IoU = 0.852174/0.997391 ≈ 0.854 (후보지만 obs21 의 1.0 이 이김)
    obs_part = np.concatenate([A[:2000], far])
    s, obs, _ = run_case(tmp, 'dup', gt, [(10, [(21, A), (22, obs_part), (23, far)])], n_frames=100)
    ob = {int(r['tracklet_id']): r for r in obs}
    check('IoU 1 인 obs21 이 매칭', ob[21]['match@20'], '1')
    check('obs22 는 매칭 안 됨', ob[22]['match@20'], '0')
    check('obs22 는 물체1 의 중복으로 표시', ob[22]['dup_of'], '1')
    check('obs22 IoU', ious(ob[22]).get(0), (2000 / 2300 * 0.98) / (2000 / 2300 + 0.98 - 2000 / 2300 * 0.98), 1e-9)
    check('obs22 는 FP (void 300/2300)', ob[22]['pred_status'], 'fp')
    check('GT 없는 obs23 은 중복 아님 · 제외', (ob[23]['dup_of'], ob[23]['pred_status']), ('0', 'excluded'))
    check('재현율은 그대로 1', s['missed']['recall'], 1.0)
    check('중복 1건', s['accuracy']['n_obs_duplicate'], 1)
    check('[E2] tide_error: obs22 dupe · obs23 bkg · 매칭된 obs21 빈 값', (ob[22]['tide_error'], ob[23]['tide_error'], ob[21]['tide_error']),
          ('dupe', 'bkg', ''))
    check('[E2] accuracy.unmatched_tide_errors', s['accuracy'].get('unmatched_tide_errors'), {'dupe': 1, 'bkg': 1})
    check('[E2] params.dup_rule 에 TIDE Dupe 와 IoU', ('TIDE' in s['params']['dup_rule'], 'IoU' in s['params']['dup_rule']), (True, True))
    check('매칭률 1/3', s['accuracy']['obs_match_rate'], 0.3333)
    s2 = S.score(tmp / 'dup', SEQ, None, max_range=1e9, gt_obj=gt, out_dir=tmp / 'dup_count', dup_policy='count',
                 surface=FakeSurface(gt), log=lambda *a: None)
    check('옛 방식(count)이면 IoU ≥ 0.5 중복도 매칭 → 매칭률 2/3', s2['accuracy']['obs_match_rate'], 0.6667)


@guarded
def test_iou_matching(tmp):
    print('A4. 3D 검출 매칭은 IoU_τ 헝가리안 — 시간 정보(직전 id 가산점) 없음')
    # 감사 A-4 이후 시간 연관은 score_mot 의 CLEAR 매칭만 맡는다. keyframe 마다 독립 매칭.
    A = plane(0, 0, 0.0, 50, 50)
    far = plane(40, 40, 0.0, 20, 25)                      # 500 셀, GT 없음
    gt = FakeGT([dict(oid=1, first=0, last=50, pts=A)])
    big, small = A, np.concatenate([A[:1500], far])       # IoU 1 vs P 0.75 · R 0.78(0..38행) → 0.585/0.945 = 13/21
    kfs = [(10, [(1, big), (2, small)]),
           (20, [(1, small), (2, big)]),
           (30, [(2, big)])]
    _, rows, _ = run_case(tmp, 'cont', gt, kfs, n_frames=100)
    m = {(int(r['kf']), int(r['tracklet_id'])): r for r in rows}
    check('kf0 id1 매칭', m[(0, 1)]['match@20'], '1')
    check('kf1 id2 매칭 (IoU 더 큼, 직전 id 와 무관)', m[(1, 2)]['match@20'], '1')
    check('kf1 id1 은 중복', m[(1, 1)]['dup_of'], '1')
    check('kf1 id1 IoU = 13/21', ious(m[(1, 1)]).get(0), 13 / 21, 1e-12)
    check('kf2 id2 매칭', m[(2, 2)]['match@20'], '1')


@guarded
def test_strict_tau(tmp):
    print('A5. strict τ (d < τ) — 정확히 20cm 떨어진 복셀은 τ 밖, float32 GT 좌표(x=10.03m)에서도')
    # 옛 기대(감사 A-6: 정확히 τ 는 '경계 포함')를 T&T evaluation.py:173-176 의 strict '<' 로 바꿨다 (score_geometry 와 같은 규칙)
    ys, zs = np.meshgrid(np.arange(30), np.arange(30))
    g64 = np.stack([np.full(ys.size, (501 + 0.5) * V), (ys.ravel() + 0.5) * V, (zs.ravel() + 0.5) * V], 1)
    gt = FakeGT([dict(oid=1, first=0, last=100, pts=g64.astype(np.float32))])
    s, ob, _ = run_case(tmp, 'eps20', gt, [(10, [(1, g64 + [0.20, 0, 0])])])
    check('+20cm: match@20 없음', ob[0]['match@20'], '0')
    check('+20cm: IoU 0 (gt_iou 빈 값)', ob[0]['gt_iou@20'], '')
    check('+20cm: 재현율 0', s['missed']['recall'], 0.0)
    s, ob, _ = run_case(tmp, 'eps18', gt, [(10, [(1, g64 + [0.18, 0, 0])])])
    check('+18cm: match@20 = 물체1 · IoU 1', (ob[0]['match@20'], float(ob[0]['match_iou@20'])), ('1', 1.0))
    check('+18cm: 20cm 이내 비율 1.0', float(ob[0]['within_tau']), 1.0)


@guarded
def test_upper_bound(tmp):
    print('A6. KD-tree 상한 — 48cm 떨어진 복셀은 τ=50cm 에서 매칭, 정확히 50cm 는 아님')
    g = plane(0, 0, 0.0, 30, 30)
    gt = FakeGT([dict(oid=1, first=0, last=100, pts=g)])
    s, ob, _ = run_case(tmp, 'ub48', gt, [(10, [(1, g + [0, 0, 0.48])])])
    check('48cm: match@50 = 물체1', ob[0]['match@50'], '1')
    check('48cm: τ50 재현율 1.0', {r['tau_cm']: r['recall'] for r in s['sensitivity']}[50], 1.0)
    check('48cm: τ20 은 매칭 없음', ob[0]['match@20'], '0')
    _, ob, _ = run_case(tmp, 'ub50', gt, [(10, [(1, g + [0, 0, 0.50])])])
    check('50cm: match@50 없음 (strict)', ob[0]['match@50'], '0')


@guarded
def test_partial_run_vis(tmp):
    print('A7. [감사 A-8] 부분 실행 — 실행 구간 밖 프레임의 가시성은 세지 않는다')
    A = plane(0, 0, 0.0, 50, 50)
    far = plane(40, 40, 0.0, 20, 25)
    gt = FakeGT([dict(oid=1, first=50, last=300, pts=A)])
    vis = dict(ep_index=np.array([0, 0]), frame=np.array([60, 200]), px_crop=np.array([300, 9000]))
    s, _, ep = run_case(tmp, 'partial', gt, [(60, [(1, far)])], n_frames=100, vis=vis)   # 실행 = 프레임 [0,100)
    v = s['visibility']
    check('탐지가능 0 (1600px 이상은 실행 밖 프레임 200 뿐)', v['n_detectable'], 0)
    check('탐지가능 재현율 null', v['recall_detectable'], None)
    check('시야 안 1 (프레임 60, 300px)', v['n_in_view'], 1)
    check('max_px_crop = 실행 안 최대 300', ep[0]['max_px_crop'], '300')
    check('vis_frames_crop = 1', ep[0]['vis_frames_crop'], '1')


@guarded
def test_empty_guards(tmp):
    print('A8. [감사 A-9] 빈 입력 — 예외 없음 · NaN 대신 null')
    A = plane(0, 0, 0.0, 50, 50)
    gt = FakeGT([dict(oid=1, first=0, last=100, pts=A)])
    s, _, _ = run_case(tmp, 'e_nodet', gt, [(10, [(1, A)])], vis=dict(ep_index=np.array([0]), frame=np.array([10]),
                                                                      px_crop=np.array([100])))
    check('(a) 탐지가능 0 → recall_detectable null', s['visibility']['recall_detectable'], None)
    check('(a) score.json 엄격 JSON', isinstance(strict_json(tmp / 'e_nodet/score.json'), dict), True)
    gt_late = FakeGT([dict(oid=1, first=1000, last=1100, pts=A)])
    vis_late = dict(ep_index=np.array([0]), frame=np.array([1050]), px_crop=np.array([5000]))
    for name, kfs in (('e_noep', [(10, [(1, A)])]), ('e_noep_nokf', [])):
        s, _, ep = run_case(tmp, name, gt_late, kfs, vis=vis_late)
        check(f'({name}) 에피소드 0', s['missed']['n_gt_episodes'], 0)
        check(f'({name}) recall null', s['missed']['recall'], None)
        check(f'({name}) recall_in_view null', s['visibility']['recall_in_view'], None)
        check(f'({name}) 엄격 JSON', isinstance(strict_json(tmp / name / 'score.json'), dict), True)
    s, _, ep = run_case(tmp, 'e_nokf', gt, [], vis=dict(ep_index=np.array([0]), frame=np.array([10]), px_crop=np.array([5000])))
    check('(keyframe 0) 재현율 0', s['missed']['recall'], 0.0)
    check('(keyframe 0) 원인 no_kf', ep[0]['miss_reason'], 'no_kf')
    check('(keyframe 0) score_kf_gt.csv 빈 파일', (tmp / 'e_nokf/score_kf_gt.csv').exists(), True)


@guarded
def test_two_episodes_one_object(tmp):
    print('A9. 같은 물체의 두 에피소드 — GT 표면이 에피소드 단위(라벨 값 ep+1)라 원인이 다른 에피소드로 새지 않는다')
    # 옛 A9 는 absorbed 기록을 에피소드 하나에만 하는지 봤다(감사 A-10). absorbed 는 폐지(→ merged)됐고 표면이 에피소드별이다.
    A = plane(0, 0, 0.0, 50, 50)
    far = plane(40, 40, 0.0, 20, 25)
    gt = FakeGT([dict(oid=1, first=0, last=50, pts=A), dict(oid=1, first=54, last=100, pts=A)])
    obs = np.concatenate([A[:600], far, plane(80, 80, 0, 20, 70)])        # 물체 600 · P 0.24 · R 0.42 → IoU 0.1008/0.5592 = 0.18
    _, _, ep = run_case(tmp, 'twoeps', gt, [(50, [(1, obs)])], n_frames=200)   # 프레임 50: ep0 [0,50] 안 · ep1 은 라벨 없음
    e = {int(r['ep_index']): r for r in ep}
    check('ep0 원인 low_iou (IoU 0.18 ∈ [0.1, 0.5], F2: 옛 하한 0.25 에서는 missed)', e[0]['miss_reason'], 'low_iou')
    check('ep1 원인 no_kf (창 [54,100] 에 keyframe 없음)', e[1]['miss_reason'], 'no_kf')
    check('ep1 표면 있는 keyframe 0', e[1]['n_kf_with_surface'], '0')


@guarded
def test_detect_credit(tmp):
    print('A10. 탐지 인정 = 그 keyframe 에 GT 가 present(라벨 5m 이내 ≥ 1600px) 일 때의 매칭만 (옛: crop 가시 px > 0)')
    A = plane(0, 0, 0.0, 50, 50)
    B = plane(3, 0, 0.0, 30, 30)
    gt = FakeGT([dict(oid=1, first=20, last=60, pts=A), dict(oid=2, first=20, last=60, pts=B)])
    surf = FakeSurface(gt, override={(30, 0): 800})                   # 프레임 30 에서 ep0 라벨 800px (present 아님)
    # 물체1: 프레임 16(에피소드 밖 → 라벨 표면 없음) · 30(표면 800px) 에만 발행 · 물체2: 프레임 40 에 발행
    kfs = [(16, [(1, A)]), (30, [(1, A)]), (40, [(2, B)])]
    s, ob, ep = run_case(tmp, 'credit', gt, kfs, surface=surf)
    e = {int(r['ep_index']): r for r in ep}
    check('ep0 탐지 안 됨', e[0]['detected'], '0')
    check('ep0 원인 missed (프레임 40 에 present 인데 관측 없음)', e[0]['miss_reason'], 'missed')
    check('ep1 탐지', e[1]['detected'], '1')
    m = {int(r['frame']): r for r in ob}
    check('프레임 16: 표면이 없어 매칭 없음 · 제외(void)', (m[16]['match@20'], m[16]['pred_status']), ('0', 'excluded'))
    check('프레임 30: 제외 매칭 (match@20 은 남음)', (m[30]['match@20'], m[30]['pred_status']), ('1', 'ignored_match'))
    check('프레임 30: detect_credit 0', m[30]['detect_credit'], '0')
    check('프레임 40: detect_credit 1', m[40]['detect_credit'], '1')
    check('ep0 제외 매칭 수 1', e[0]['n_matched_obs_no_credit'], '1')


@guarded
def test_obs_columns(tmp):
    print('A11. 관측별 열 — gt_counts@20(물체별, 가장 가까운 표면) · gt_vox@20(에피소드별) · gt_iou@20(에피소드별 IoU)')
    A = plane(0, 0, 0.0, 50, 50)
    B = plane(3, 0, 0.0, 30, 30)
    gt = FakeGT([dict(oid=1, first=0, last=50, pts=A), dict(oid=2, first=0, last=60, pts=B)])
    s, ob, _ = run_case(tmp, 'counts', gt, [(20, [(12, np.concatenate([B, A[:600]])), (13, plane(40, 40, 0, 10, 10))])])
    check('params.counts_column', s['params'].get('counts_column'), 'gt_counts@20')
    check('params.iou_column', s['params'].get('iou_column'), 'gt_iou@20')
    check('params.vox_column', s['params'].get('vox_column'), 'gt_vox@20')
    m = {int(r['tracklet_id']): r for r in ob}
    check('obs12 gt_counts@20', m[12].get('gt_counts@20'), '1:600;2:900')
    check('obs12 gt_vox@20', m[12].get('gt_vox@20'), '0:600;1:900')
    iu = ious(m[12])
    check('obs12 gt_iou@20 에피소드 = [0, 1]', sorted(iu), [0, 1])
    check('obs12 gt_iou@20[0] = 0.168/0.652', iu.get(0), 0.168 / 0.652, 1e-12)
    check('obs12 gt_iou@20[1] = 0.6', iu.get(1), 0.6, 1e-12)
    check('obs13 (GT 없음) 빈 값', (m[13].get('gt_counts@20'), m[13].get('gt_vox@20'), m[13].get('gt_iou@20')), ('', '', ''))


@guarded
def test_fragment(tmp):
    print('F1. 조각 사례 — 큰 조각(P 0.85) vs 순수한 작은 조각(P 1.0)')
    # 감사(apartment kf29 / ep78): share 최대 헝가리안이 192복셀 share 1.0 조각을 2403복셀 share 0.87 조각보다 골랐다.
    G = cells(0, 0, 0, 100, 1)                           # GT 선분 100셀
    small = G[:10]                                       # P 1 · R 19/100 → IoU 0.19
    big = np.concatenate([G[:85], cells(0, 50, 0, 15, 1)])   # P 0.85 · R 94/100 → IoU 0.799/0.991
    gt = FakeGT([dict(oid=7, first=0, last=100, pts=G)])
    s, ob, _ = run_case(tmp, 'frag', gt, [(10, [(1, small), (2, big)])])
    m = {int(r['tracklet_id']): r for r in ob}
    check('share(옛 top_frac): 작은 조각 1.0 · 큰 조각 0.85', (float(m[1]['top_frac']), float(m[2]['top_frac'])), (1.0, 0.85))
    check('큰 조각 매칭', m[2]['match@20'], '7')
    check('큰 조각 IoU = 0.799/0.991', float(m[2]['match_iou@20']), 0.799 / 0.991, 1e-9)
    check('큰 조각 P·R', (float(m[2]['match_P@20']), float(m[2]['match_R@20'])), (0.85, 0.94))
    # [E2] 옛 기대 dup_of 7 (P_τ 최대 GT 가 매칭됨) → TIDE Dupe 는 IoU ≥ 0.5 를 요구 (quantify.py:251). IoU 0.19 ∈ [0.1, 0.5] → Loc (F2)
    check('작은 조각 매칭 안 됨 · 중복 아님 (IoU 0.19) · FP · TIDE loc', (m[1]['match@20'], m[1]['dup_of'], m[1]['pred_status'], m[1]['tide_error']),
          ('0', '0', 'fp', 'loc'))
    kg = read_csv(tmp / 'frag/score_kf_gt.csv')
    check('score_kf_gt: tp · 매칭 관측 = 큰 조각', (kg[0]['status'], kg[0]['matched_tracklet_id']), ('tp', '2'))


@guarded
def test_ignored_match(tmp):
    print('F2. 제외 매칭 — present 아닌 GT(라벨 1000px)에 매칭된 예측은 TP 도 FP 도 아니다')
    A = plane(0, 0, 0.0, 50, 50)
    gt = FakeGT([dict(oid=1, first=0, last=100, pts=A)])
    vis = dict(ep_index=np.array([0]), frame=np.array([10]), px_crop=np.array([5000]))   # 가시성으로는 채점 대상
    s, ob, ep = run_case(tmp, 'ign', gt, [(10, [(1, A)])], vis=vis, surface=FakeSurface(gt, px=1000))
    check('match@20 = 1 · 상태 ignored_match', (ob[0]['match@20'], ob[0]['pred_status']), ('1', 'ignored_match'))
    check('detect_credit 0', ob[0]['detect_credit'], '0')
    check('탐지 안 됨 · 원인 no_kf (present keyframe 없음)', (ep[0]['detected'], ep[0]['miss_reason']), ('0', 'no_kf'))
    a = s['accuracy']
    check('TP 0 · 제외 매칭 1 · FP 0 · 제외 0', (a['n_obs_tp'], a['n_obs_ignored_match'], a['n_obs_fp'], a['n_obs_excluded']),
          (0, 1, 0, 0))
    check('n_obs_matched_no_credit 1', a['n_obs_matched_no_credit'], 1)
    kg = read_csv(tmp / 'ign/score_kf_gt.csv')
    check('score_kf_gt: present 0 · 상태 ignored_match', (kg[0]['present'], kg[0]['status']), ('0', 'ignored_match'))


@guarded
def test_presence_boundary(tmp):
    print('F3. present 경계 — 라벨 px 1599 는 아님, 1600 은 present (sam.AREA_MIN 256 proto 칸 × 2.5²)')
    A = plane(0, 0, 0.0, 50, 50)
    gt = FakeGT([dict(oid=1, first=0, last=100, pts=A)])
    surf = FakeSurface(gt, override={(10, 0): 1599, (20, 0): 1600})
    s, ob, ep = run_case(tmp, 'pres', gt, [(10, [(1, A)]), (20, [(1, A)])], surface=surf)
    m = {int(r['frame']): r for r in ob}
    check('1599px: ignored_match', m[10]['pred_status'], 'ignored_match')
    check('1600px: tp', m[20]['pred_status'], 'tp')
    check('탐지 인정 1 · 제외 매칭 1', (ep[0]['n_matched_obs'], ep[0]['n_matched_obs_no_credit']), ('1', '1'))
    check('present keyframe 1', ep[0]['n_kf_present'], '1')
    check('keyframe_gt.n_present 1', s['keyframe_gt']['n_present'], 1)


@guarded
def test_fp_excluded(tmp):
    print('F4. 미매칭 예측 FP / 제외 — (void + 무시 GT 위 복셀)/|P| > 0.5 면 제외 (panopticapi evaluation.py:161)')
    A = plane(0, 0, 0.0, 50, 50)                          # ep0 present
    Cn = plane(0, 3, 0.0, 50, 50)                         # ep1 present 아님 (800px)
    far = plane(40, 40, 0.0, 50, 50)
    gt = FakeGT([dict(oid=1, first=0, last=100, pts=A), dict(oid=2, first=0, last=100, pts=Cn)])
    surf = FakeSurface(gt, override={(10, 1): 800})
    obs = [(1, A),                                                   # tp
           (2, np.concatenate([A[:50], far[:50]])),                   # void 50/100 = 0.5 → FP
           (3, np.concatenate([A[:49], far[:51]])),                   # void 0.51 → 제외 void
           (4, np.concatenate([A[:40], Cn[:30], far[:30]])),          # void 30 + 무시 GT 30 = 0.6 → 제외 void+ignored_gt
           (5, np.concatenate([A[:50], Cn[:50]])),                    # 무시 GT 0.5 → FP
           (6, np.concatenate([Cn[:60], far[:40]]))]                  # 무시 GT 0.6 · void 0.4 → 제외 ignored_gt
    s, ob, _ = run_case(tmp, 'fpx', gt, [(10, obs)], surface=surf)
    m = {int(r['tracklet_id']): r for r in ob}
    check('상태', [m[i]['pred_status'] for i in range(1, 7)], ['tp', 'fp', 'excluded', 'excluded', 'fp', 'excluded'])
    check('제외 사유', [m[i]['excluded_reason'] for i in (3, 4, 6)], ['void', 'void+ignored_gt', 'ignored_gt'])
    check('ignore_frac@20 (obs2 · obs3 · obs5)', [float(m[i]['ignore_frac@20']) for i in (2, 3, 5)], [0.5, 0.51, 0.5])
    a = s['accuracy']
    check('TP 1 · FP 2 · 제외 3', (a['n_obs_tp'], a['n_obs_fp'], a['n_obs_excluded']), (1, 2, 3))
    check('제외 사유 집계', a['excluded_reasons'], {'void': 1, 'void+ignored_gt': 1, 'ignored_gt': 1})


@guarded
def test_miss_reasons(tmp):
    print('F5. present GT 의 미검출 원인 — score_2d 규칙을 τ 허용 복셀 집합에 (merged · split · low_iou · missed)')
    M1 = cells(0, 0, 0, 30, 30); M2 = cells(50, 0, 0, 30, 30)
    Sg = cells(0, 100, 0, 90, 10)
    L48 = cells(0, 200, 0, 50, 10); L25 = cells(0, 300, 0, 40, 10); L24 = cells(0, 400, 0, 50, 10); E = cells(0, 500, 0, 20, 20)
    gt = FakeGT([dict(oid=i + 1, first=0, last=100, pts=p) for i, p in enumerate([M1, M2, Sg, L48, L25, L24, E])])
    obs = [(1, np.concatenate([M1, M2, cells(0, 1000, 0, 100, 1)])),      # P(M1) = 900/1900 · R 1 → 둘 다 merged
           (2, cells(0, 100, 0, 20, 10)), (3, cells(35, 100, 0, 20, 10)), (4, cells(70, 100, 0, 20, 10)),   # split (합집합 1.0)
           (5, cells(0, 200, 0, 15, 10)),                                  # R 24/50 = 0.48 → low_iou
           (6, cells(0, 300, 0, 1, 10)),                                   # R 10/40 = 0.25 → low_iou (경계 포함)
           (7, cells(0, 400, 0, 3, 10))]                                   # R 12/50 = 0.24 → low_iou (F2: 하한 0.1)
    s, _, ep = run_case(tmp, 'reasons', gt, [(10, obs)])
    kg = {int(r['ep_index']): r for r in read_csv(tmp / 'reasons/score_kf_gt.csv')}
    check('keyframe 상태', [kg[e]['status'] for e in range(7)], ['merged', 'merged', 'split', 'low_iou', 'low_iou', 'low_iou', 'missed'])
    check('best_iou (L48 · L25 · L24 · E)', [float(kg[e]['best_iou']) for e in (3, 4, 5, 6)], [0.48, 0.25, 0.24, 0.0])
    check('split 조각 3 · 덮음 1.0', (kg[2]['n_frags'], float(kg[2]['split_cover'])), ('3', 1.0))
    e = {int(r['ep_index']): r for r in ep}
    check('트랙 원인 = keyframe 상태', [e[i]['miss_reason'] for i in range(7)],
          ['merged', 'merged', 'split', 'low_iou', 'low_iou', 'low_iou', 'missed'])
    check('n_absorbed_obs (deprecated = merged keyframe 수)', (e[0]['n_absorbed_obs'], e[2]['n_absorbed_obs']), ('1', '0'))
    check('miss_reasons 집계', s['missed']['miss_reasons'], {'merged': 2, 'split': 1, 'low_iou': 3, 'missed': 1})


@guarded
def test_track_reason(tmp):
    print('F6. 트랙 원인 — present keyframe 없음 → no_kf (라벨 < 1600px 뿐이어도) · 여러 keyframe 이면 최빈값')
    L = cells(0, 0, 0, 50, 10)
    Q = cells(0, 200, 0, 20, 20)
    gt = FakeGT([dict(oid=1, first=0, last=100, pts=L), dict(oid=2, first=0, last=100, pts=Q)])
    surf = FakeSurface(gt, override={(f, 1): 1000 for f in (10, 20, 30)})   # ep1 은 라벨 1000px 뿐
    vis = dict(ep_index=np.array([0, 1]), frame=np.array([10, 10]), px_crop=np.array([5000, 5000]))
    kfs = [(10, [(1, cells(0, 0, 0, 15, 10))]),     # ep0 low_iou (0.48)
           (20, [(1, cells(0, 0, 0, 15, 10))]),     # low_iou
           (30, [(1, cells(0, 100, 0, 3, 10))])]    # missed (2m 떨어진 예측 → IoU 0 < 0.1)
    s, _, ep = run_case(tmp, 'treason', gt, kfs, vis=vis, surface=surf)
    e = {int(r['ep_index']): r for r in ep}
    check('ep0 원인 low_iou (low_iou 2 · missed 1)', e[0]['miss_reason'], 'low_iou')
    check('ep0 keyframe 수 (low_iou 2 · missed 1)', (e[0]['n_kf_low_iou'], e[0]['n_kf_missed']), ('2', '1'))
    check('ep1 원인 no_kf (채점 대상이지만 라벨 present keyframe 없음)', e[1]['miss_reason'], 'no_kf')
    check('탐지가능 원인 집계', s['visibility']['miss_reasons_detectable'], {'low_iou': 1, 'no_kf': 1})
    check('params.reason_tie_order', s['params'].get('reason_tie_order'), ['merged', 'split', 'low_iou', 'missed'])
    check('params.reason_mapping', s['params'].get('reason_mapping'), {'absorbed': 'merged', 'not_detected': 'missed'})


@guarded
def test_exclusive_merge(tmp):
    print('F8. [E1] 배타적 덮음 — 벽 예측 20cm 안의 얇은 문틀은 merged 가 아니다 (감사 fix_D fig kf58 ep118/119/120)')
    W = cells(0, 0, 0, 60, 60); Tr = cells(0, 60, 5, 60, 3)
    gt = FakeGT([dict(oid=1, first=0, last=100, pts=W), dict(oid=2, first=0, last=100, pts=Tr)])
    s, ob, ep = run_case(tmp, 'xmerge', gt, [(10, [(1, W.copy())]), (20, [(1, np.concatenate([W, Tr]))])])
    kg = {(int(r['frame']), int(r['ep_index'])): r for r in read_csv(tmp / 'xmerge/score_kf_gt.csv')}
    check('프레임 10 (벽만): 벽 tp · 문틀 low_iou (옛: merged, 최고 IoU 0.133)', (kg[(10, 0)]['status'], kg[(10, 1)]['status']), ('tp', 'low_iou'))
    check('프레임 20 (벽 ∪ 문틀): 문틀 merged · merged_obs = 그 관측', (kg[(20, 1)]['status'], kg[(20, 1)]['merged_obs']), ('merged', '1'))
    check('params.merge_split_coverage 에 배타(nearest) 규칙', 'nearest' in s['params'].get('merge_split_coverage', ''), True)


def gt_as_prediction(gt, shift=(0, 0, 0), skip_every=0):
    by_frame = {}
    skipped = 0
    for i in range(len(gt.tid)):
        if skip_every and i % skip_every == 0:
            skipped += 1; continue
        by_frame.setdefault(int(gt.last[i]), []).append((int(gt.tid[i]), gt.points(i) + np.array(shift)))
    return sorted(by_frame.items()), skipped


def real_labels_dir(seq='apartment_s1_00h'):
    import gt_labels_2d as GL
    for c in (os.environ.get('FB_TEST_LABELS'), RUNS / 'gt_labels_2d' / f'uHumans2_{seq}', RUNS / '_audit/fix_B_2d' / seq / 'labels'):
        try:
            if c and json.loads((Path(c) / 'meta.json').read_text()).get('version') == GL.LABEL_VERSION:
                return Path(c)
        except (OSError, ValueError):
            pass
    return None


def perfect_kfs(labels, n=6, shift=(0, 0, 0), drop_every=0):
    """실제 keyframe n 개에서 present GT 마다 '라벨 픽셀을 frontend 격자(192×256, 픽셀 floor(2.5i)·floor(2.5j))로 뽑아
    역투영한 점' 을 관측 하나로 → (kfs, present 쌍 수, 뺀 present 쌍 수). drop_every: present GT 순서 i % drop_every == 0 이면 뺀다."""
    import gt_surface as GS
    ls = GS.LabelSurface(SEQ, labels)
    frames = sorted(int(p.stem) for p in labels.glob('*.png'))
    frames = frames[::max(1, len(frames) // n)][:n]
    gi = np.floor(np.arange(192) * 2.5).astype(int); gj = np.floor(np.arange(256) * 2.5).astype(int)
    GV, GU = np.meshgrid(gi, gj, indexing='ij')
    kfs, n_present, n_drop, i = [], 0, 0, 0
    for fr in frames:
        lab, dep, K, R, t = ls.frame_geometry(fr)
        surf = ls.at(fr)
        Lg, Zg = lab[GV, GU], dep[GV, GU]
        cam = GS.backproject(GU, GV, Zg, K)
        ok = (Zg > 0) & (np.linalg.norm(cam, axis=-1) <= 5.0)
        obs = []
        for e in sorted(surf):
            if surf[e].px < 1600:
                continue
            n_present += 1; i += 1
            if drop_every and i % drop_every == 0:
                n_drop += 1; continue
            p = cam[ok & (Lg == e + 1)] @ R.T + t
            obs.append((100000 * len(kfs) + e, p + np.array(shift)))
        kfs.append((fr, obs))
    return kfs, n_present, n_drop


def test_real(tmp):
    labels = real_labels_dir()
    if labels is None:
        print(f'  SKIP R. 최신 버전 GT 라벨 폴더 없음 (REAL_RUNS={RUNS})')
        return
    import gt_labels_2d as GL
    gt = S.GT(GTH5)
    nfr = S.UH2Sequence(SEQ).n_frames
    print(f'R1. 완벽 예측 (라벨 {labels})')
    kfs, n_pres, _ = perfect_kfs(labels)
    write_run(tmp / 'perfect', kfs, nfr)
    s = S.score(tmp / 'perfect', SEQ, GTH5, gt_obj=gt, labels_dir=labels, out_dir=tmp / 'perfect', log=lambda *a: None)
    kg = [r for r in read_csv(tmp / 'perfect/score_kf_gt.csv') if r['present'] == '1']
    check(f'present 쌍 {n_pres} 전부 tp', (len(kg), sum(r['status'] == 'tp' for r in kg)), (n_pres, n_pres))
    iou = np.array([float(r['match_iou']) for r in kg])
    print(f'      IoU_τ min {iou.min():.4f} · median {np.median(iou):.4f}')
    check('IoU_τ 최솟값 ≥ 0.95 (≈ 1)', bool(iou.min() >= 0.95), True)
    check('keyframe_gt.recall 1.0', s['keyframe_gt']['recall'], 1.0)
    check('params.labels_version', s['params'].get('labels_version'), GL.LABEL_VERSION)
    check('params.labels_meta_sha1', s['params'].get('labels_meta_sha1'), hashlib.sha1((labels / 'meta.json').read_bytes()).hexdigest())
    print('R2. +100m 이동')
    kfs, _, _ = perfect_kfs(labels, shift=(100, 0, 0))
    write_run(tmp / 'far', kfs, nfr)
    s = S.score(tmp / 'far', SEQ, GTH5, gt_obj=gt, labels_dir=labels, out_dir=tmp / 'far', max_range=1e9, log=lambda *a: None)
    kg = [r for r in read_csv(tmp / 'far/score_kf_gt.csv') if r['present'] == '1']
    check('매칭 0', s['accuracy']['n_obs_matched'], 0)
    check('present GT 전부 missed', {r['status'] for r in kg}, {'missed'})
    print('R3. present GT 3개 중 1개꼴로 빼기')
    kfs, n_pres, n_drop = perfect_kfs(labels, drop_every=3)
    write_run(tmp / 'drop', kfs, nfr)
    s = S.score(tmp / 'drop', SEQ, GTH5, gt_obj=gt, labels_dir=labels, out_dir=tmp / 'drop', log=lambda *a: None)
    check(f'검출 재현율 = ({n_pres} − {n_drop}) / {n_pres}', s['keyframe_gt']['recall'], round((n_pres - n_drop) / n_pres, 4))


if __name__ == '__main__':
    pos = [a for a in sys.argv[1:] if not a.startswith('--')]
    with tempfile.TemporaryDirectory(dir=pos[0] if pos else None) as d:
        for fn in (test_synthetic, test_duplicates, test_iou_matching, test_strict_tau, test_upper_bound, test_partial_run_vis,
                   test_empty_guards, test_two_episodes_one_object, test_detect_credit, test_obs_columns, test_fragment,
                   test_ignored_match, test_presence_boundary, test_fp_excluded, test_miss_reasons, test_track_reason,
                   test_exclusive_merge):
            fn(Path(d))
        if '--synthetic-only' not in sys.argv:
            guarded(test_real)(Path(d))
    print('\n결과:', '전부 통과' if not FAILS else f'실패 {len(FAILS)}개 {FAILS}')
    sys.exit(1 if FAILS else 0)
