#!/usr/bin/env python3
"""gt_surface.py 자체 검증 — keyframe 하나의 GT 표면 (라벨 픽셀 역투영 · 5m · 2cm 복셀).

G1 가짜 시퀀스 (depth 3m 평면 · 회전+이동 pose)
     ep0 80×80px 영역 (3m)          → 표면 있음, px 6398 (주점 두 픽셀은 ep3·ep4), 복셀 = 직접 역투영한 값
     ep1 (6m 영역, 5m 밖)           → 표면 없음
     ep2 (depth 0 영역)             → 표면 없음
     ep3 주점(principal point) 한 픽셀, depth 5.0 → 광선 길이 정확히 5.0m → 포함 (≤, score_geometry.py:156 과 같음)
     ep4 그 옆 픽셀, depth 5.0      → 광선 길이 > 5m → 제외
G2 score_geometry.score_sequence 가 쓰는 GT 복셀(geometry_metrics 의 gt 인자)과 복셀 단위로 같다 — 가짜 시퀀스
G3 같은 비교를 실제 apartment_s1_00h keyframe 3개에서 (라벨: FB_TEST_LABELS → runs/frontend_eval/gt_labels_2d
     → _audit/fix_B_2d (paths.REAL_RUNS 기준), meta.json version 이 gt_labels_2d.LABEL_VERSION 과 같은 첫 폴더. 없으면 SKIP)
G4 labels_meta: meta.json 의 version 과 sha1 · 라벨 PNG 가 없으면 FileNotFoundError (조용히 빈 표면으로 두지 않음)
G5 prefetch_poses (한 번에 부른 pose 캐시) 가 프레임마다 부른 pose 와 비트 단위로 같고, 표면도 복셀 단위로 같다 (실제 keyframe 12개)
"""
import csv
import hashlib
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import REAL_RUNS as RUNS, seq_dir  # noqa: E402  (실제 데이터 — FB_RUNS 와 무관, 수정 ⑤ E4)

FAILS = []


def check(name, got, want):
    ok = bool(np.all(np.asarray(got) == np.asarray(want))) if isinstance(want, (list, tuple, np.ndarray)) else got == want
    print(f'  {"OK " if ok else "FAIL"} {name}: got={got} want={want}')
    if not ok:
        FAILS.append(name)


Q = Rotation.from_euler('xyz', [0.3, -0.2, 0.9]).as_quat()
T = np.array([[1.234, -2.345, 0.567]])


class FakeSeq:
    Kfull = np.array([[415.692, 0, 360.0], [0, 415.692, 240.0], [0, 0, 1]])

    def __init__(self, *_):
        self.stamps_ns = np.arange(5)

    def camera_info(self, cam='left_cam'):
        return {'K': self.Kfull.copy()}

    def depth(self, i):
        d = np.full((480, 720), 3.0, np.float32)
        d[0:40, 40:80] = 6.0                     # crop 열 0:40 → 6m
        d[440:480, 640:680] = 0.0                # crop 열 600:640 → depth 없음
        d[240, 360] = 5.0; d[240, 361] = 5.0     # crop (240,320) = 주점 · (240,321)
        return d

    def camera_poses(self, stamps, cam='left_cam'):
        return T.copy(), Q[None].copy()


def fake_label():
    L = np.zeros((480, 640), np.uint16)
    L[200:280, 280:360] = 1
    L[0:40, 0:40] = 2
    L[440:480, 600:640] = 3
    L[240, 320] = 4
    L[240, 321] = 5
    return L


def _GS():
    import gt_surface as GS  # noqa: E402  (고치기 전 코드에는 없는 모듈)
    return GS


def test_fake():
    """G1 가짜 시퀀스 역투영 · 5m 경계"""
    GS = _GS()
    with tempfile.TemporaryDirectory() as d:
        lab = Path(d)
        cv2.imwrite(str(lab / '000000.png'), fake_label())
        (lab / 'meta.json').write_text(json.dumps(dict(version='test')))
        ls = GS.LabelSurface(None, lab, max_range=5.0, seq=FakeSeq())
        s = ls.at(0)
        check('표면 있는 에피소드 = {0, 3}', sorted(s), [0, 3])
        check('ep0 px = 80×80 − 2 (주점 두 픽셀을 ep3·ep4 가 덮음) = 6398', s[0].px, 6398)
        check('ep3 px = 1 (광선 길이 정확히 5m 포함)', s.get(3) and s[3].px, 1)
        # 직접 계산: crop K (cx 320), R·t, floor(p/0.02)
        R = Rotation.from_quat(Q).as_matrix(); f = 415.692
        vv, uu = np.nonzero(fake_label() == 1)
        cam = np.stack([(uu - 320.0) / f * 3.0, (vv - 240.0) / f * 3.0, np.full(len(uu), 3.0)], 1)
        w = cam @ R.T + T[0]
        k = np.unique(np.floor(w / 0.02).astype(np.int64), axis=0)
        check('ep0 복셀 수', len(s[0].vox), len(k))
        check('ep0 복셀 = 직접 계산', np.allclose(s[0].vox, (k + 0.5) * 0.02, atol=1e-12, rtol=0), True)
        check('ep3 복셀 = 카메라 앞 5m 한 점', np.allclose(s[3].vox, (np.floor((np.array([0, 0, 5.0]) @ R.T + T[0]) / 0.02) + 0.5) * 0.02), True)
        check('voxelize 는 score_geometry.voxelize 와 같은 값', np.array_equal(GS.voxelize(w), __import__('score_geometry').voxelize(w)), True)


class Capture:
    """score_geometry.geometry_metrics 를 감싸 (frame, ep) 별 gt 인자를 모은다. 파일은 고치지 않는다."""

    def __init__(self, SG):
        self.SG, self.orig, self.got, self.ctx = SG, SG.geometry_metrics, [], None

    def __enter__(self):
        def wrap(pred, gt, *a, **k):
            self.got.append(np.array(gt, copy=True))
            return self.orig(pred, gt, *a, **k)
        self.SG.geometry_metrics = wrap
        return self

    def __exit__(self, *a):
        self.SG.geometry_metrics = self.orig


def run_geometry(SG, run, seqd, labels, rows, out):
    """rows = [(kf, frame, obs, ep)] 를 score_observations.csv 로 쓰고 score_geometry 를 돌려, 관측마다 gt 복셀을 돌려준다."""
    with open(run / 'score_observations.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['kf', 'frame', 'obs', 'tracklet_id', 'match@20', 'ep_index', 'detect_credit'])
        for kf, fr, o, e in rows:
            w.writerow([kf, fr, o, o, 1, e, 1])
    with Capture(SG) as cap:
        SG.score_sequence(run, seqd, labels, out_dir=out, log=lambda *_: None)
    return cap.got[0::2] if len(cap.got) == 2 * len(rows) else cap.got   # 관측마다 (실제, 완벽 예측 상한) 두 번 호출


def test_geometry_identity_fake():
    """G2 score_geometry GT 복셀과 같다 (가짜 시퀀스)"""
    GS = _GS()
    import score_geometry as SG
    orig_seq = SG.UH2Sequence
    with tempfile.TemporaryDirectory() as d:
        d = Path(d); lab = d / 'labels'; run = d / 'run'; lab.mkdir(); run.mkdir()
        cv2.imwrite(str(lab / '000000.png'), fake_label())
        R = Rotation.from_quat(Q).as_matrix()
        pts = (np.array([[0, 0, 2.0]] * 5) @ R.T + T[0]).astype(np.float32)   # 카메라 앞 2m — 5m 이내 예측 점
        with h5py.File(run / 'frontend_output.h5', 'w') as f:
            f['obs/points_start'] = np.array([0, 5]); f['obs/points_num'] = np.array([5, 0]); f['points'] = pts
        try:
            SG.UH2Sequence = FakeSeq
            got = run_geometry(SG, run, d / 'seq', lab, [(0, 0, 0, 0), (0, 0, 0, 3)], d / 'out')
        finally:
            SG.UH2Sequence = orig_seq
        s = GS.LabelSurface(None, lab, max_range=5.0, seq=FakeSeq()).at(0)
        check('score_geometry 가 채점한 관측 2', len(got), 2)
        check('ep0 복셀 동일 (array_equal)', len(got) > 0 and np.array_equal(got[0], s[0].vox), True)
        check('ep3 복셀 동일', len(got) > 1 and np.array_equal(got[1], s[3].vox), True)


def real_labels_dir(seq='apartment_s1_00h'):
    import gt_labels_2d as GL
    for c in (os.environ.get('FB_TEST_LABELS'), RUNS / 'gt_labels_2d' / f'uHumans2_{seq}', RUNS / '_audit/fix_B_2d' / seq / 'labels'):
        try:
            if c and json.loads((Path(c) / 'meta.json').read_text()).get('version') == GL.LABEL_VERSION:
                return Path(c)
        except (OSError, ValueError):
            pass
    return None


def test_geometry_identity_real():
    """G3 score_geometry GT 복셀과 같다 (실제 keyframe 3개)"""
    GS = _GS()
    import score_geometry as SG
    seq = 'apartment_s1_00h'
    labels = real_labels_dir(seq)
    run_real = RUNS / f'uHumans2_{seq}' / 'frontend_output.h5'
    if labels is None or not run_real.exists():
        print('  SKIP: 최신 버전 라벨 또는 frontend_output.h5 없음')
        return
    print(f'  라벨: {labels}')
    ls = GS.LabelSurface(seq_dir(seq), labels)
    with h5py.File(run_real, 'r') as f:
        kf_frame = f['kf/frame_idx'][:]; o0 = f['kf/obs_start'][:]; no = f['kf/n_obs'][:]; pn = f['obs/points_num'][:]
    rows, want = [], []
    for k in (5, 60, 150):
        fr = int(kf_frame[k]); s = ls.at(fr)
        o = int(o0[k] + np.argmax(pn[o0[k]:o0[k] + no[k]]))            # 점이 가장 많은 관측 (5m 이내 점이 있어야 채점된다)
        for e in sorted(s, key=lambda e: -s[e].px)[:3]:
            rows.append((k, fr, o, e)); want.append(s[e].vox)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d); run = d / 'run'; run.mkdir()
        os.symlink(run_real, run / 'frontend_output.h5')
        got = run_geometry(SG, run, seq_dir(seq), labels, rows, d / 'out')
    check(f'채점된 (keyframe, GT) {len(rows)}', len(got), len(rows))
    same = [np.array_equal(g, w) for g, w in zip(got, want)]
    check(f'복셀 동일 {sum(same)}/{len(rows)} (복셀 수 {[len(w) for w in want]})', all(same) and len(same) == len(rows), True)


def test_meta():
    """G4 labels_meta · 라벨 없음"""
    GS = _GS()
    with tempfile.TemporaryDirectory() as d:
        lab = Path(d)
        raw = json.dumps(dict(version='v-test', seq='x'), indent=2).encode()
        (lab / 'meta.json').write_bytes(raw)
        m = GS.labels_meta(lab)
        check('version', m.get('version'), 'v-test')
        check('meta_sha1', m.get('meta_sha1'), hashlib.sha1(raw).hexdigest())
        check('labels_dir', m.get('labels_dir'), str(lab))
        ls = GS.LabelSurface(None, lab, seq=FakeSeq())
        check('LabelSurface.meta() = labels_meta', ls.meta(), m)
        try:
            ls.at(0)
            check('라벨 PNG 없음 → FileNotFoundError', 'no error', 'FileNotFoundError')
        except FileNotFoundError:
            check('라벨 PNG 없음 → FileNotFoundError', 'FileNotFoundError', 'FileNotFoundError')


def test_prefetch_poses():
    """G5 prefetch_poses = 프레임마다 부른 pose (비트 단위) · 표면 동일"""
    GS = _GS()
    seq = 'apartment_s1_00h'
    labels = real_labels_dir(seq)
    run_real = RUNS / f'uHumans2_{seq}' / 'frontend_output.h5'
    if labels is None or not run_real.exists():
        print('  SKIP: 최신 버전 라벨 또는 frontend_output.h5 없음')
        return
    with h5py.File(run_real, 'r') as f:
        frames = [int(x) for x in f['kf/frame_idx'][::17]]
    a = GS.LabelSurface(seq_dir(seq), labels)
    b = GS.LabelSurface(seq_dir(seq), labels).prefetch_poses(frames)
    same_pose, same_vox = 0, 0
    for fr in frames:
        t, q = a.seq.camera_poses(a.seq.stamps_ns[[fr]], cam='left_cam')
        same_pose += int(np.array_equal(t, b.poses[fr][0]) and np.array_equal(q, b.poses[fr][1]))
        sa, sb = a.at(fr), b.at(fr)
        same_vox += int(sorted(sa) == sorted(sb) and all(np.array_equal(sa[e].vox, sb[e].vox) and sa[e].px == sb[e].px for e in sa))
    check(f'pose 비트 동일 {same_pose}/{len(frames)}', same_pose, len(frames))
    check(f'표면 동일 {same_vox}/{len(frames)}', same_vox, len(frames))


def run():
    for fn in (test_fake, test_geometry_identity_fake, test_geometry_identity_real, test_meta, test_prefetch_poses):
        print(f'[{fn.__name__}] {fn.__doc__.strip()}')
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc(limit=4)
            print(f'  FAIL {fn.__name__}: 예외 {type(e).__name__}: {e}')
            FAILS.append(f'{fn.__name__} 예외')
    return FAILS


if __name__ == '__main__':
    run()
    print('\n결과:', '전부 통과' if not FAILS else f'실패 {len(FAILS)}개 {FAILS}')
    sys.exit(1 if FAILS else 0)
