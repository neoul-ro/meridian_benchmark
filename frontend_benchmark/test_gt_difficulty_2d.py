#!/usr/bin/env python3
"""gt_difficulty_2d.py 자체 검증 — level_of 경계, 잘림 실루엣(카메라 뒤·캔버스 밖·같은 물체 합집합), 사람 테두리.

카메라 = 월드 (R=I, t=0), 640x480 crop 기준 K: fx=fy=415.692, cx=320, cy=240.

1. level_of 경계 (KITTI evaluate_object.cpp:44-46,413 — 높이는 '<= MIN_HEIGHT 이면 무시', 잘림·가림은 '> MAX 이면 무시')
     최소 폭 40 → easy 아님 · 41 → easy · 25 → 등급 밖 · 26 → moderate · 잘림 0.15 → easy 유지 · 0.16 → moderate
2. 사람(잘림 -1)인데 라벨이 화면 테두리에 닿음 → easy 제외 (결함 13)
3. silhouette: 화면 안 패치만 → 잘림 0 / + 같은 수의 카메라 뒤 점 → 잘림 > 0.5 / + 3배 캔버스 밖 점 → 잘림 > 0.5 (결함 7)
4. frame_rows: z=3 평면 x∈[-4.3,0], 활성 에피소드 e0 는 x∈[-2.3,0] 만 봤고 같은 물체의 e1 이 x∈[-4.3,-2.0] 을 봄.
     화면 왼쪽 끝 x=-2.31 → 실제 잘림 ≈ 2.0/4.3 ≈ 0.46 → hard. 에피소드 점군만 쓰면 ≈ 0 → easy (결함 4)
     같은 프레임 사람 라벨이 위쪽 테두리에 닿음 → level 1 (결함 13)
"""
import sys
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gt_difficulty_2d as D  # noqa: E402

FAILS = []
K = np.array([[415.692, 0, 320.0], [0, 415.692, 240.0], [0, 0, 1]])
R0, T0 = np.eye(3), np.zeros(3)


def check(name, got, want, tol=None):
    try:
        ok = abs(got - want) <= tol if tol is not None else got == want
    except TypeError:
        ok = False
    print(f'  {"OK " if ok else "FAIL"} {name}: got={got} want={want}')
    if not ok:
        FAILS.append(name)


def grid(x0, x1, y0, y1, z):
    xs = np.arange(x0, x1, 0.02) + 0.01; ys = np.arange(y0, y1, 0.02) + 0.01
    X, Y = np.meshgrid(xs, ys)
    return np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1)


def test_level_of_boundaries():
    """level_of 경계 — KITTI 높이 strict >, 잘림·가림 ≤"""
    check('min_dim 40 → easy 아님(1)', D.level_of(40, 0.0, 0.0), 1)
    check('min_dim 41 → easy(0)', D.level_of(41, 0.0, 0.0), 0)
    check('min_dim 25 → 등급 밖(3)', D.level_of(25, 0.0, 0.0), 3)
    check('min_dim 26 → moderate(1)', D.level_of(26, 0.0, 0.0), 1)
    check('잘림 0.15 → easy 유지', D.level_of(41, 0.15, 0.20), 0)
    check('잘림 0.16 → moderate', D.level_of(41, 0.16, 0.0), 1)
    check('가림 0.5 · 잘림 0.3 → moderate', D.level_of(26, 0.30, 0.50), 1)
    check('잘림 0.5 · 가림 0.9 → hard', D.level_of(26, 0.50, 0.90), 2)
    check('잘림 0.51 → 등급 밖', D.level_of(26, 0.51, 0.0), 3)


def test_human_border():
    """사람(잘림·가림 -1): 테두리에 닿으면 easy 제외"""
    check('사람 · 테두리 닿음 → moderate', D.level_of(100, -1.0, -1.0, at_border=True), 1)
    check('사람 · 테두리 안 닿음 → easy', D.level_of(100, -1.0, -1.0, at_border=False), 0)
    check('정적(잘림 계산됨)은 at_border 무관', D.level_of(100, 0.0, 0.0, at_border=True), 0)


def trunc_of(pts):
    i, tot = D.silhouette(pts, R0, T0, K)
    return 1.0 - i / tot if tot else 0.0


def test_silhouette_offscreen():
    """silhouette — 카메라 뒤·3배 캔버스 밖 점을 '화면 밖'으로 센다"""
    patch = grid(-0.3, 0.3, -0.3, 0.3, 3.0)
    check('화면 안 패치 → 잘림 0', trunc_of(patch), 0.0, 1e-9)
    behind = patch.copy(); behind[:, 2] = -1.0
    t = trunc_of(np.concatenate([patch, behind]))
    check('+ 카메라 뒤 같은 수 → 잘림 > 0.5', t > 0.5, True); print(f'     (잘림 {t:.3f})')
    off = grid(29.7, 30.3, -0.3, 0.3, 2.0)                  # u ≈ 6555 → 3배 캔버스(-640..1280) 밖
    t = trunc_of(np.concatenate([patch, off]))
    check('+ 3배 캔버스 밖 같은 수 → 잘림 > 0.5', t > 0.5, True); print(f'     (잘림 {t:.3f})')
    edge = grid(1.9, 2.7, -0.3, 0.3, 3.0)                   # u 599..690 → 오른쪽 끝에 걸침 (캔버스 안)
    t = trunc_of(edge)
    check('오른쪽 끝에 걸친 패치 → 잘림 0.2~0.8 (캔버스 안은 그대로)', 0.2 < t < 0.8, True); print(f'     (잘림 {t:.3f})')
    t = trunc_of(behind)
    check('전부 카메라 뒤 → 잘림 1', t, 1.0, 1e-9)


class FakeSeq:
    def __init__(self, dep):
        self.dep = dep; self.stamps_ns = np.arange(10)

    def depth(self, i):
        return self.dep

    def camera_poses(self, stamps, cam='left_cam'):
        return np.zeros((1, 3)), np.array([[0.0, 0.0, 0.0, 1.0]])


def test_frame_rows_union_and_human():
    """frame_rows — 같은 gt_object_id 합집합 실루엣, 사람 테두리"""
    e0 = grid(-2.3, 0.0, -0.5, 0.5, 3.0); e1 = grid(-4.3, -2.0, -0.5, 0.5, 3.0)
    with tempfile.TemporaryDirectory() as d:
        lab = np.zeros((480, 640), np.uint16)
        v0, v1 = int(240 - 0.5 * 415.692 / 3), int(240 + 0.5 * 415.692 / 3)
        lab[v0:v1 + 1, 0:321] = 1                     # e0 의 보이는 부분
        lab[0:60, 500:560] = 3                        # 사람 ep2, 위쪽 테두리에 닿음
        cv2.imwrite(str(Path(d) / '000003.png'), lab)
        dep = np.full((480, 720), 3.0, np.float32)
        G = D.G; G.clear()
        G['seq'] = FakeSeq(dep); G['K'] = K; G['labels'] = Path(d)
        G['tid'] = np.arange(3); G['oid'] = np.array([5, 5, 9]); G['human'] = np.array([False, False, True])
        G['pts'] = {0: e0, 1: e1, 2: grid(0.5, 0.9, -1.5, -1.0, 3.0)}
        G['obj_pts'] = {5: np.concatenate([e0, e1])}
        rows = {r[1]: r for r in D.frame_rows(3)}
        t = rows[0][2]
        check('e0 잘림 = 같은 물체 합집합 기준 ≈ 0.46', 0.40 <= t <= 0.52, True); print(f'     (잘림 {t:.3f})')
        check('e0 등급 hard(2)', rows[0][7], 2)
        check('사람 테두리 → level 1', rows[2][7], 1)


def run():
    for fn in (test_level_of_boundaries, test_human_border, test_silhouette_offscreen, test_frame_rows_union_and_human):
        print(f'[{fn.__name__}] {fn.__doc__.strip()}')
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc(limit=3)
            print(f'  FAIL {fn.__name__}: 예외 {type(e).__name__}: {e}')
            FAILS.append(fn.__name__)
    return FAILS


def main():
    run()
    print('\n결과(난이도):', '전부 통과' if not FAILS else f'실패 {len(FAILS)}개 {FAILS}')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
