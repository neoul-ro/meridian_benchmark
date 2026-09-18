#!/usr/bin/env python3
"""score_geometry.py 자체 검증 — 답을 아는 점군.

1. 같은 점군          → 정확도·완전도·Chamfer 0 · P/R/F 전부 1
2. 법선으로 8cm 이동   → 정확도·완전도·Chamfer 8cm · τ5 에서 0, τ10·20 에서 1
3. 예측이 GT 의 왼쪽 절반만 (1m×1m 평면, 2cm 격자) → 정확도 0 · 완전도 = 오른쪽 절반의 평균 거리
     F@τ 는 Tanks and Temples 절차(evaluation.py:77,83 두 점군 τ/2 복셀 다운샘플, :173-176 d < τ).
     τ=20: 10cm 복셀 (Open3D 식 원점 = 최소점 - 5cm) → x 중심 GT 0.03,0.11,…,0.91,0.98 (11열) · 예측 0.03,…,0.41,0.48 (6열)
     → GT 중 예측 0.48 에서 20cm 미만 = x ≤ 0.61 인 7열 → R@20 = 7/11, P@20 = 1
     (예전 기대값 0.70 은 다운샘플 없이 '≤ τ' 로 센 비표준 값이라 바꿨다)
4. 빈 입력 → None
5. 경계: 거리가 정확히 τ=10cm → 표준은 d < τ 라 P=R=F=0 (결함 11, 감사 사례 6)
6. 끝단 (가짜 시퀀스): 라벨이 빈 매칭 관측은 건너뛴 수로 JSON 에 남고 (감사 C 결함 10),
     완벽 예측 상한(GT 라벨 픽셀을 frontend 격자 밀도로 뽑은 예측)이 JSON 에 기록된다 (결함 5). 기존 키·열 유지.
7. detect_credit=0 인 매칭 관측(26/09/17 결정 ④ 이후 뜻: present 아닌 GT 에 매칭 = 제외 매칭)은 채점하지 않고
     skipped_obs['ignored_match'] 로 센다 [수정 ⑤]. 옛 키 outside_detect_window 는 같은 값의 deprecated 별칭 — n_skipped 에 두 번 넣지 않는다.
     gt_label_empty 검사보다 먼저 세서 두 사유가 겹치지 않는다. 열이 없는 옛 CSV 는 1.
8. [수정 ⑤] 매칭 열은 score.json params.tau_m 에서 (match@<τcm>) — match@20 고정이 아니다. score.json 이 없으면 옛 기본 20cm.
9. [수정 ⑤] GT 표면은 gt_surface.LabelSurface (pose 한 번에 캐시). 옛 구현(프레임마다 camera_poses · mgrid 역투영 · np.unique 복셀)을
     이 파일에 사본으로 두고 geometry_metrics 에 들어가는 네 배열(예측 복셀 · GT 복셀 · 예측 원점 · GT 원점)과 완벽 예측 원점이
     비트 단위로 같은지 본다 — 가짜 시퀀스 · 실제 apartment_s1_00h keyframe 12개 (실제 데이터: paths.REAL_RUNS, 없으면 SKIP)
"""
import csv
import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_geometry as S  # noqa: E402

FAILS = []


def check(name, got, want, tol=1e-6):
    try:
        ok = got is not None and (abs(got - want) <= tol if tol is not None else got == want)
    except TypeError:
        ok = False
    print(f'  {"OK " if ok else "FAIL"} {name}: got={got} want={want}')
    if not ok:
        FAILS.append(name)


def plane(nx, ny, z=0.0):
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny))
    return np.stack([(ix.ravel() + 0.5) * 0.02, (iy.ravel() + 0.5) * 0.02, np.full(ix.size, z)], 1)


def test_basic():
    """1-4. 같은 점군 · 8cm 이동 · 왼쪽 절반 · 빈 입력"""
    G = plane(50, 50)
    print('1. 같은 점군')
    m = S.geometry_metrics(G, G)
    for k in ('accuracy_cm', 'completeness_cm', 'chamfer_l1_cm'):
        check(k, m[k], 0.0)
    for t in (5, 10, 20):
        check(f'F@{t}', m[f'F@{t}'], 1.0)
    print('2. 8cm 이동')
    m = S.geometry_metrics(G + [0, 0, 0.08], G)
    check('accuracy 8cm', m['accuracy_cm'], 8.0, 1e-6); check('completeness 8cm', m['completeness_cm'], 8.0, 1e-6)
    check('chamfer 8cm', m['chamfer_l1_cm'], 8.0, 1e-6)
    check('F@5 = 0', m['F@5'], 0.0); check('F@10 = 1', m['F@10'], 1.0); check('F@20 = 1', m['F@20'], 1.0)
    print('3. 왼쪽 절반만 예측')
    half = G[G[:, 0] < 0.5]
    m = S.geometry_metrics(half, G)
    d = np.clip(G[:, 0] - (0.5 - 0.01), 0, None)                # 오른쪽 점 → 가장 가까운 예측 열(x=0.49)까지
    check('accuracy 0', m['accuracy_cm'], 0.0)
    check('completeness = 거리 평균', m['completeness_cm'], float(d.mean() * 100), 1e-6)
    check('P@20 = 1', m['P@20'], 1.0)
    check('R@20 = 7/11 (T&T τ/2 다운샘플)', m['R@20'], 7 / 11, 1e-9)
    print('4. 빈 입력')
    check('예측 없음 → None 이면 통과', 0.0 if S.geometry_metrics(np.zeros((0, 3)), G) is None else 1.0, 0.0)


def test_strict_tau():
    """5. 거리 정확히 τ → 포함하지 않는다 (T&T evaluation.py:173-176, NICE-SLAM completion_ratio 'd < th')"""
    gt = np.array([[0.0, 0.0, 0.0]]); pred = np.array([[0.0, 0.0, 0.1]])
    m = S.geometry_metrics(pred, gt)
    check('거리 = 0.1 (부동소수 확인)', float(np.linalg.norm(pred - gt)), 0.1, 0.0)
    check('P@10 = 0', m['P@10'], 0.0); check('R@10 = 0', m['R@10'], 0.0); check('F@10 = 0', m['F@10'], 0.0)
    check('F@20 = 1', m['F@20'], 1.0)


def test_voxel_down_sample():
    """Open3D VoxelDownSample 과 같은 격자·평균 (PointCloud.cpp: 원점 = min_bound - v/2, floor, 평균)"""
    p = np.array([[0.00, 0.0, 0.0], [0.04, 0.0, 0.0], [0.06, 0.0, 0.0], [0.30, 0.0, 0.0]])
    q = S.voxel_down_sample(p, 0.1)                            # 원점 -0.05 → 칸 0: 0.00,0.04 · 칸 1: 0.06 · 칸 3: 0.30
    q = q[np.argsort(q[:, 0])]
    check('칸 수 3', len(q), 3, None)
    check('칸 0 평균 x = 0.02', q[0, 0], 0.02, 1e-12); check('칸 1 x = 0.06', q[1, 0], 0.06, 1e-12)


class FakeSeq:
    Kfull = np.array([[415.692, 0, 360.0], [0, 415.692, 240.0], [0, 0, 1]])

    def __init__(self):
        self.stamps_ns = np.arange(5)

    def camera_info(self, cam='left_cam'):
        return {'K': self.Kfull.copy()}

    def depth(self, i):
        return np.full((480, 720), 3.0, np.float32)

    def camera_poses(self, stamps, cam='left_cam'):
        return np.zeros((1, 3)), np.array([[0.0, 0.0, 0.0, 1.0]])


OLD_JSON_KEYS = ['run', 'params', 'n_obs', 'summary', 'seconds']
OLD_PARAM_KEYS = ['voxel_m', 'taus_m', 'max_range_m', 'stride', 'gt', 'chamfer']
OLD_CSV = ('kf,frame,obs,tracklet_id,ep_index,n_pred_vox,n_gt_vox,accuracy_cm,completeness_cm,chamfer_l1_cm,'
           'P@5,R@5,F@5,P@10,R@10,F@10,P@20,R@20,F@20').split(',')


def _fake_run(d, obs_rows, header=('kf', 'frame', 'obs', 'tracklet_id', 'match@20', 'ep_index')):
    """z=3 평면 · crop rows 200:280 cols 280:360 = ep0 라벨 (ep1 라벨 없음). obs0 = ep0 라벨을 frontend 격자로 뽑은 점,
    obs1 = 그 일부를 50cm 띄운 점, obs2 = obs0 의 앞 10점. obs_rows 를 score_observations.csv 로 쓰고 채점 → JSON."""
    import cv2
    import h5py
    K = FakeSeq.Kfull.copy(); K[0, 2] -= 40
    run = d / 'run'; lab = d / 'labels'; run.mkdir(); lab.mkdir()
    L = np.zeros((480, 640), np.uint16); L[200:280, 280:360] = 1
    cv2.imwrite(str(lab / '000000.png'), L)
    gi = np.arange(192); gj = np.arange(256)                                # frontend 격자 표본 (sam.py nearest)
    vv, uu = np.meshgrid(np.floor(gi * 2.5).astype(int), np.floor(gj * 2.5).astype(int), indexing='ij')
    sel = L[vv, uu] == 1
    p0 = np.stack([(uu[sel] - K[0, 2]) / K[0, 0] * 3.0, (vv[sel] - K[1, 2]) / K[1, 1] * 3.0, np.full(sel.sum(), 3.0)], 1)
    p1 = p0[:50] + [0, 0, 0.5]
    pts = np.concatenate([p0, p1, p0[:10]]).astype(np.float32)
    with h5py.File(run / 'frontend_output.h5', 'w') as f:
        f['obs/points_start'] = np.array([0, len(p0), len(p0) + len(p1)])
        f['obs/points_num'] = np.array([len(p0), len(p1), 10]); f['points'] = pts
    with open(run / 'score_observations.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(obs_rows)
    S.UH2Sequence = lambda _: FakeSeq()                                     # 데이터셋 대신 가짜 시퀀스
    out = S.score_sequence(run, d / 'seq', lab, out_dir=run, log=lambda *_: None)
    return out, json.loads((run / 'score_geometry.json').read_text()), run


def test_end_to_end():
    """6. 끝단 — 건너뛴 매칭 관측 수 · 완벽 예측 상한 · 기존 키/열"""
    with tempfile.TemporaryDirectory() as d:
        out, j, run = _fake_run(Path(d), [[0, 0, 0, 0, 1, 0], [0, 0, 1, 1, 1, 1], [0, 0, 2, 2, 0, '']])
        check('채점된 관측 1', j['n_obs'], 1, None)
        check('매칭 관측 2', j.get('n_matched_obs'), 2, None)
        check('건너뜀: GT 라벨 빈 관측 1', j.get('skipped_obs', {}).get('gt_label_empty'), 1, None)
        check('건너뜀 합계 1', j.get('n_skipped'), 1, None)
        check('detect_credit 열 없음(옛 CSV) → 1 로 봄: ignored_match 0', j.get('skipped_obs', {}).get('ignored_match'), 0, None)
        check('deprecated 별칭 outside_detect_window 0', j.get('skipped_obs', {}).get('outside_detect_window'), 0, None)
        pu = j.get('perfect_upper_bound') or {}
        check('완벽 예측 상한 관측 수 1', pu.get('n_obs'), 1, None)
        row = next(csv.DictReader(open(run / 'score_geometry.csv')))
        check('완벽 예측 = 이 관측(같은 격자 표본) → F@20 같음', (pu.get('summary') or {}).get('F@20', {}).get('median'),
              float(row['F@20']), 1e-4)
        check('완벽 예측 정확도 < 0.5cm', (pu.get('summary') or {}).get('accuracy_cm', {}).get('median', 9) < 0.5, True, None)
        check('JSON 기존 키 유지', [k for k in OLD_JSON_KEYS if k not in j], [], None)
        check('params 기존 키 유지', [k for k in OLD_PARAM_KEYS if k not in j['params']], [], None)
        check('CSV 기존 열 유지', [c for c in OLD_CSV if c not in row], [], None)
        check('반환값 = JSON', out['n_obs'], j['n_obs'], None)


def test_detect_credit():
    """7. detect_credit=0 (제외 매칭: present 아닌 GT 에 매칭) → ignored_match 로 건너뜀, gt_label_empty 보다 먼저 센다"""
    hdr = ('kf', 'frame', 'obs', 'tracklet_id', 'match@20', 'ep_index', 'detect_credit')
    with tempfile.TemporaryDirectory() as d:
        out, j, run = _fake_run(Path(d), [[0, 0, 0, 0, 1, 0, 1],      # 채점
                                          [0, 0, 1, 1, 1, 1, 0],      # credit 0 이면서 라벨도 빔 → ignored_match 로만
                                          [0, 0, 2, 2, 1, 0, 0]],     # credit 0 → ignored_match
                                header=hdr)
        sk = j.get('skipped_obs', {})
        check('채점된 관측 1', j['n_obs'], 1, None)
        check('매칭 관측 3', j.get('n_matched_obs'), 3, None)
        check('ignored_match 2', sk.get('ignored_match'), 2, None)
        check('deprecated 별칭 outside_detect_window = 같은 값 2', sk.get('outside_detect_window'), 2, None)
        check('gt_label_empty 0 (겹쳐 세지 않음)', sk.get('gt_label_empty'), 0, None)
        check('건너뜀 합계 2 (별칭을 두 번 더하지 않음)', j.get('n_skipped'), 2, None)
        check('SKIP_REASONS 첫 사유 = ignored_match', S.SKIP_REASONS[0], 'ignored_match', None)
        check('params.deprecated_skip_aliases', (j.get('params') or {}).get('deprecated_skip_aliases'), {'outside_detect_window': 'ignored_match'}, None)
        check('채점된 관측은 obs 0', [int(r['obs']) for r in csv.DictReader(open(run / 'score_geometry.csv'))], [0], None)


def test_match_column_from_params():
    """8. 매칭 열 = score.json params.tau_m (match@<τcm>)"""
    hdr = ('kf', 'frame', 'obs', 'tracklet_id', 'match@10', 'match@20', 'ep_index', 'detect_credit')
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # τ = 10cm 로 채점한 결과: match@10 에만 매칭, match@20 은 0 (τ 를 바꿨을 때 옛 코드는 매칭 0 으로 읽었다)
        rows = [[0, 0, 0, 0, 1, 0, 0, 1], [0, 0, 2, 2, 1, 0, 0, 1]]
        out, j, run = _fake_run(d, rows, header=hdr)                    # score.json 없이 한 번
        (run / 'score.json').write_text(json.dumps({'params': {'tau_m': 0.1}}))
        j2 = S.score_sequence(run, d / 'seq', d / 'labels', out_dir=run / 'o2', log=lambda *_: None)
        check('score.json 없음 → 옛 기본 match@20 (매칭 0)', j.get('n_matched_obs'), 0, None)
        check('score.json τ 10cm → match@10 을 읽음 (매칭 2)', j2.get('n_matched_obs'), 2, None)
        check('params.match_column = match@10', (j2.get('params') or {}).get('match_column'), 'match@10', None)
        check('채점된 관측 2', j2['n_obs'], 2, None)


def _reference_gt(seq, labels, fr, e, max_range=5.0, stride=1):
    """수정 ⑤ 전 score_geometry.score_sequence 의 GT 계산 사본 (프레임마다 camera_poses · mgrid · np.unique) — 대조 기준.
    → (GT 원점, GT 2cm 복셀, 완벽 예측 원점)"""
    import cv2
    from scipy.spatial.transform import Rotation
    K = seq.camera_info()['K'].copy(); K[0, 2] -= 40
    lab = cv2.imread(str(Path(labels) / f'{fr:06d}.png'), cv2.IMREAD_UNCHANGED)
    dep = seq.depth(fr)[:, 40:40 + 640]
    t, q = seq.camera_poses(seq.stamps_ns[[fr]], cam='left_cam'); R = Rotation.from_quat(q[0]).as_matrix()
    vv, uu = np.mgrid[0:480:stride, 0:640:stride]
    gi = np.floor(np.arange(192) * 2.5).astype(int); gj = np.floor(np.arange(256) * 2.5).astype(int)
    GV, GU = np.meshgrid(gi, gj, indexing='ij')

    def backproject(u, v, Z):
        return np.stack([(u - K[0, 2]) / K[0, 0] * Z, (v - K[1, 2]) / K[1, 1] * Z, Z], -1)
    L = lab[::stride, ::stride]; Z = dep[::stride, ::stride]
    cam_all = backproject(uu, vv, Z)
    ok_all = (Z > 0) & (np.linalg.norm(cam_all, axis=-1) <= max_range)
    Lg = lab[GV, GU]; Zg = dep[GV, GU]
    cam_g = backproject(GU, GV, Zg)
    ok_g = (Zg > 0) & (np.linalg.norm(cam_g, axis=-1) <= max_range)
    gt_raw = cam_all[ok_all & (L == e + 1)] @ R.T + t[0]
    gt = (np.unique(np.floor(gt_raw / 0.02).astype(np.int64), axis=0) + 0.5) * 0.02
    pp = cam_g[ok_g & (Lg == e + 1)] @ R.T + t[0]
    return gt_raw, gt, pp


class _Capture:
    def __init__(self):
        self.orig, self.calls = S.geometry_metrics, []

    def __enter__(self):
        def wrap(pred, gt, *a, **k):
            self.calls.append((np.array(pred, copy=True), np.array(gt, copy=True), np.array(k.get('pred_raw'), copy=True),
                               np.array(k.get('gt_raw'), copy=True)))
            return self.orig(pred, gt, *a, **k)
        S.geometry_metrics = wrap
        return self

    def __exit__(self, *a):
        S.geometry_metrics = self.orig


def _compare_with_reference(run, seqd, labels, jobs, seq):
    """jobs = [(kf, frame, obs, ep)] → (같은 관측 수, 전체 관측 수). 관측마다 geometry_metrics 두 번(실제 · 완벽 예측)."""
    with open(run / 'score_observations.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['kf', 'frame', 'obs', 'tracklet_id', 'match@20', 'ep_index', 'detect_credit'])
        for kf, fr, o, e in jobs:
            w.writerow([kf, fr, o, o, 1, e, 1])
    with _Capture() as cap:
        S.score_sequence(run, seqd, labels, out_dir=run / 'out', log=lambda *_: None)
    same = 0
    for n, (kf, fr, o, e) in enumerate(jobs):
        if 2 * n + 1 >= len(cap.calls):
            break
        gt_raw, gt, pp = _reference_gt(seq, labels, fr, e)
        (_, g1, _, r1), (p2, g2, pr2, r2) = cap.calls[2 * n], cap.calls[2 * n + 1]
        same += int(np.array_equal(g1, gt) and np.array_equal(r1, gt_raw) and np.array_equal(g2, gt) and np.array_equal(r2, gt_raw)
                    and np.array_equal(pr2, pp))
    return same, len(jobs)


def test_gt_surface_reference_fake():
    """9a. GT 표면 = 옛 구현과 비트 단위로 같다 (가짜 시퀀스)"""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _fake_run(d, [[0, 0, 0, 0, 1, 0]])                                     # 폴더·h5·라벨 만들기
        same, n = _compare_with_reference(d / 'run', d / 'seq', d / 'labels', [(0, 0, 0, 0), (0, 0, 2, 0)], FakeSeq())
        check('옛 GT 계산과 배열 동일 (관측 2)', same, n, None)
        check('params.gt_surface = gt_surface.LabelSurface', 'gt_surface' in json.loads((d / 'run/out/score_geometry.json').read_text())['params'].get('gt', ''), True, None)


def test_gt_surface_reference_real():
    """9b. GT 표면 = 옛 구현과 비트 단위로 같다 (실제 apartment_s1_00h keyframe 12개)"""
    import os
    import h5py
    import gt_labels_2d as GL
    import paths as PP
    seq = 'apartment_s1_00h'
    h5 = PP.REAL_RUNS / f'uHumans2_{seq}' / 'frontend_output.h5'
    labels = None
    for c in (os.environ.get('FB_TEST_LABELS'), PP.REAL_RUNS / 'gt_labels_2d' / f'uHumans2_{seq}', PP.REAL_RUNS / '_audit/fix_B_2d' / seq / 'labels'):
        try:
            if c and json.loads((Path(c) / 'meta.json').read_text()).get('version') == GL.LABEL_VERSION:
                labels = Path(c); break
        except (OSError, ValueError):
            pass
    if labels is None or not h5.exists() or not PP.seq_dir(seq).exists():
        print(f'  SKIP 실제 데이터 없음 (최신 라벨 · {h5} · 데이터셋) — FB_REAL_RUNS 확인')
        return
    import gt_surface as GS
    from meridian_benchmark.uhumans2 import UH2Sequence
    S.UH2Sequence = UH2Sequence
    ls = GS.LabelSurface(PP.seq_dir(seq), labels)
    with h5py.File(h5, 'r') as f:
        kf_frame = f['kf/frame_idx'][:]; o0 = f['kf/obs_start'][:]; no = f['kf/n_obs'][:]; pn = f['obs/points_num'][:]
    jobs = []
    for k in range(0, len(kf_frame), 17):
        fr = int(kf_frame[k]); s = ls.at(fr)
        if not no[k]:
            continue
        o = int(o0[k] + np.argmax(pn[o0[k]:o0[k] + no[k]]))
        for e in sorted(s, key=lambda e: -s[e].px)[:3]:
            jobs.append((k, fr, o, e))
    with tempfile.TemporaryDirectory() as d:
        run = Path(d) / 'run'; run.mkdir()
        os.symlink(h5, run / 'frontend_output.h5')
        same, n = _compare_with_reference(run, PP.seq_dir(seq), labels, jobs, UH2Sequence(PP.seq_dir(seq)))
    print(f'     라벨 {labels} · keyframe {len({j[1] for j in jobs})}개 · (keyframe, GT) {n}')
    check(f'옛 GT 계산과 배열 동일 {same}/{n}', same, n, None)


def run():
    for fn in (test_basic, test_strict_tau, test_voxel_down_sample, test_end_to_end, test_detect_credit,
               test_match_column_from_params, test_gt_surface_reference_fake, test_gt_surface_reference_real):
        print(f'[{fn.__name__}] {fn.__doc__.strip().splitlines()[0]}')
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc(limit=3)
            print(f'  FAIL {fn.__name__}: 예외 {type(e).__name__}: {e}')
            FAILS.append(fn.__name__)
    return FAILS


def main():
    run()
    print('\n결과:', '전부 통과' if not FAILS else f'실패 {len(FAILS)}개 {FAILS}')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
