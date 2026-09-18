#!/usr/bin/env python3
"""gt_labels_2d.py 자체 검증 — 합성 seg/depth 배열로 label_frame 을 돌린다 (파일·h5 없이 가짜 시퀀스).

카메라 = 월드 (R=I, t=0), 720x480, fx=fy=415.692, cx=360, cy=240. 평면 물체는 z 고정 사각형.
같은 seg 색 C1 이 한 연결 성분으로 이어진 세 영역
  A-near  rows 200:280 cols 300:380  z=4.0  물체 A(oid 10) 활성 에피소드 eA0 의 점군
          그 안 rows 250:256 cols 330:336 은 depth 0 (검증 불가)
  A-far   rows 200:280 cols 380:430  z=6.0  (5m 밖) 물체 A 의 다른(비활성) 에피소드 eA1 이 5m 안에서 본 표면
  B-far   rows 200:280 cols 430:480  z=6.5  (5m 밖) 같은 prefab 의 다른 물체 B(oid 11) — GT 점군이 여기 없음
색 C2  D  rows 330:390 cols 300:400 z=3  활성 에피소드 eD0 의 점군은 cols 300:325 뿐 → 성분 표본의 ~1/3 만 hit
색 C3  E  rows 60:120 cols 300:350 · F rows 60:120 cols 350:400 z=3  둘 다 활성 → 픽셀 단위로 나뉨 (원래 동작)

기대 (감사 B 결함 3 · 감사 C 결함 2)
  A-near → eA0 · A-far → eA0 (같은 물체 전 에피소드 합집합 6cm 이내) · B-far → void (예전: eA0 로 합쳐짐)
  depth 0 픽셀 → void (예전: 성분 채우기로 eA0)
  D: hit/표본 < 0.5 → 성분 채우기 안 함, 검증된 픽셀만 (cols 340:400 void, 예전: 전부 eD0)
  E/F: 서로 다른 라벨로 나뉨
"""
import sys
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # 채점 코드는 한 칸 위 (frontend_benchmark/)
import gt_labels_2d as L  # noqa: E402

FAILS = []
K = np.array([[415.692, 0, 360.0], [0, 415.692, 240.0], [0, 0, 1]])
C1, C2, C3 = 0xaabbcc, 0x112233, 0x445566


def check(name, got, want, tol=None):
    try:
        ok = abs(got - want) <= tol if tol is not None else got == want
    except TypeError:
        ok = False
    print(f'  {"OK " if ok else "FAIL"} {name}: got={got} want={want}')
    if not ok:
        FAILS.append(name)


class FakeSeq:
    def __init__(self, seg, dep):
        self.seg, self.dep = seg, dep
        self.stamps_ns = np.arange(200)

    def seg_packed(self, i):
        return self.seg

    def depth(self, i):
        return self.dep

    def camera_poses(self, stamps, cam='left_cam'):
        return np.zeros((1, 3)), np.array([[0.0, 0.0, 0.0, 1.0]])


def plane_pts(r0, r1, c0, c1, z):
    vv, uu = np.mgrid[r0:r1, c0:c1]
    return np.stack([(uu - K[0, 2]) / K[0, 0] * z, (vv - K[1, 2]) / K[1, 1] * z, np.full(uu.shape, z)], -1).reshape(-1, 3)


def build_scene():
    seg = np.zeros((480, 720), np.int64); dep = np.full((480, 720), 10.0, np.float32)
    for (r0, r1, c0, c1, z, c) in [(200, 280, 300, 380, 4.0, C1), (200, 280, 380, 430, 6.0, C1), (200, 280, 430, 480, 6.5, C1),
                                   (330, 390, 300, 400, 3.0, C2), (60, 120, 300, 400, 3.0, C3)]:
        seg[r0:r1, c0:c1] = c; dep[r0:r1, c0:c1] = z
    dep[250:256, 330:336] = 0.0
    eps = [  # (oid, color, first, last, points)
        (10, C1, 0, 10, plane_pts(200, 280, 300, 380, 4.0)),      # 0 eA0 활성
        (10, C1, 100, 110, plane_pts(200, 280, 380, 430, 6.0)),   # 1 eA1 비활성 (A 의 먼 부분을 다른 때 5m 안에서 봄)
        (11, C1, 100, 110, plane_pts(0, 40, 0, 40, 20.0)),        # 2 eB0 비활성, 전혀 다른 곳
        (12, C2, 0, 10, plane_pts(330, 390, 300, 325, 3.0)),      # 3 eD0 활성, D 의 왼쪽 1/4 만
        (13, C3, 0, 10, plane_pts(60, 120, 300, 350, 3.0)),       # 4 eE 활성
        (14, C3, 0, 10, plane_pts(60, 120, 350, 400, 3.0)),       # 5 eF 활성
    ]
    G = L.G
    G.clear()
    G['seq'] = FakeSeq(seg, dep); G['K'] = K
    G['tid'] = np.arange(len(eps)); G['oid'] = np.array([e[0] for e in eps])
    G['color'] = np.array([e[1] for e in eps], np.int64)
    G['first'] = np.array([e[2] for e in eps]); G['last'] = np.array([e[3] for e in eps])
    G['trees'] = {i: cKDTree(e[4]) for i, e in enumerate(eps)}
    G['obj_trees'] = {oid: cKDTree(np.concatenate([e[4] for e in eps if e[0] == oid])) for oid in {e[0] for e in eps}}
    G.update(stride=2, max_range=5.0, tol=0.06, dominant=0.9, support_min=0.5)


def test_label_frame():
    """합성 장면 label_frame — 5m 밖 합쳐짐 · depth 0 · 낮은 지지 성분 · 픽셀 분할"""
    build_scene()
    lab, counts = L.label_frame(5)
    c = lambda col: col - L.CROP            # noqa: E731  원본 열 → 640 crop 열
    check('출력 크기 480x640 uint16', (lab.shape, lab.dtype), ((480, 640), np.dtype(np.uint16)))
    a_near = lab[200:280, c(300):c(380)].copy(); a_near[50:56, 30:36] = 1   # depth 0 구멍은 따로 본다
    check('A-near 전부 eA0(=1)', bool((a_near == 1).all()), True)
    check('depth 0 픽셀은 void', int(lab[250:256, c(330):c(336)].max()), 0)
    check('A-far (5m 밖, 같은 물체 합집합 6cm 이내) → eA0', float((lab[200:280, c(380):c(430)] == 1).mean()), 1.0)
    check('B-far (5m 밖, 다른 물체) → void', int((lab[200:280, c(430):c(480)] > 0).sum()), 0)
    check('D 검증된 왼쪽 cols 300:320 → eD0(=4)', float((lab[330:390, c(300):c(320)] == 4).mean()), 1.0)
    check('D 지지 < 0.5 → 오른쪽 cols 340:400 void', int((lab[330:390, c(340):c(400)] > 0).sum()), 0)
    check('E cols 300:340 → eE(=5)', float((lab[60:120, c(300):c(340)] == 5).mean()), 1.0)
    check('F cols 360:400 → eF(=6)', float((lab[60:120, c(360):c(400)] == 6).mean()), 1.0)
    check('counts 는 crop 라벨 픽셀 수와 같다', sum(counts.values()), int((lab > 0).sum()))


def test_outdated_labels():
    """옛 알고리즘 라벨 폴더(meta.json version 다름/없음)는 build 가 건너뛰지 않고 다시 만든다"""
    import json
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        check('meta.json 없음 → 새로 만들 대상 아님(빈 폴더)', L.labels_outdated(d), False)
        (d / '000000.png').write_bytes(b'x')
        check('PNG 있는데 meta.json 없음 → 옛 라벨', L.labels_outdated(d), True)
        (d / 'meta.json').write_text(json.dumps(dict(stride=2)))
        check('version 없는 meta.json → 옛 라벨', L.labels_outdated(d), True)
        (d / 'meta.json').write_text(json.dumps(dict(version=L.LABEL_VERSION)))
        check('현재 version → 그대로 씀', L.labels_outdated(d), False)


def run():
    for fn in (test_label_frame, test_outdated_labels):
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
    print('\n결과(GT 라벨):', '전부 통과' if not FAILS else f'실패 {len(FAILS)}개 {FAILS}')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
