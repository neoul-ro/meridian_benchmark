#!/usr/bin/env python3
"""keyframe 하나의 GT 표면 — 3D 매칭(score_frontend · score_mot)의 GT (26/09/17 사용자 결정 ④).

정의 (수정 ⑤ 부터 score_geometry.score_sequence 도 이 모듈의 frame() · raw() 를 쓴다. 아래 식은 옮기기 전 score_geometry 의
     GT 원점 계산과 같은 식·같은 연산 순서 — test_score_geometry 9a·9b 가 옛 구현 사본과 배열을 비트 단위로 대조한다)
  라벨     gt_labels_2d.py 가 만든 640x480 uint16 PNG, 값 = ep_index + 1 (0 = void)
  역투영   그 keyframe 의 depth(가운데 640 crop) · K(cx − 40) · GT pose(left_cam) 로 모든 라벨 픽셀(stride 1)
           score_geometry: `K[0, 2] -= CROP` · `vv, uu = np.mgrid[0:480:stride, 0:640:stride]` · `backproject(u, v, Z)` ·
           `dep = seq.depth(fr)[:, CROP:CROP + 640]` · `seq.camera_poses(seq.stamps_ns[[fr]], cam='left_cam')` ·
           `gt_raw = cam_all[ok_all & (L == e + 1)] @ R.T + t[0]`
  범위     depth > 0 이고 카메라 광선 길이 ‖cam‖ ≤ max_range(5m) — score_geometry 의 `ok_all = (Z > 0) & (norm ≤ max_range)` 와 같은 '≤'
  복셀     2cm 셀 중심 (floor(p/v) + 0.5)·v, 중복 제거 — score_geometry.voxelize 와 같은 값·같은 순서.
           test_gt_surface.py 가 score_geometry 가 실제로 쓰는 GT 복셀과 array_equal 로 같음을 확인한다 (가짜·실제 keyframe).
  px       위 범위를 통과한 라벨 픽셀 수 — present 판정(≥ 1600px = sam.AREA_MIN 256 proto 칸 × 2.5²)에 쓴다.
출력  at(frame) → {ep_index: Surface(vox (M,3) float64, px int)}. 표면 복셀이 0 인 GT 는 넣지 않는다.
      frame(frame, stride) → FrameGT (라벨 · depth · K · R · t · stride 표본 라벨/역투영/범위 마스크) · raw(fg, ep) → GT 원점 (M,3)
      (score_geometry 가 T&T F@τ 에 원점을 쓰고, 완벽 예측 상한에 라벨·depth 를 쓴다)
      라벨 PNG 가 없으면 FileNotFoundError — GT 를 모르는 keyframe 을 '물체 없음' 으로 채점하지 않는다.
pose     UH2Sequence.camera_poses 는 부를 때마다 odom 전체로 Slerp 를 새로 만들어 keyframe 당 ~70ms 가 든다 (프로파일: 시간의 40%).
         prefetch_poses(frames) 는 한 번에 여러 stamp 를 넣어 부르고 결과를 캐시한다. 한 번에 부른 값과 프레임마다 부른 값
         (score_geometry 방식)이 비트 단위로 같음을 apartment_s1_00h 204 · office_s1_06h 1032 keyframe 전부에서 확인했고
         (_audit/fix_D_match3d/pose_bitwise.py → pose_bitwise.out), test_gt_surface.py G5 가 실제 keyframe 에서 다시 확인한다.
"""
import hashlib
import json
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

CROP = 40
VOX = 0.02
H, W = 480, 640
Surface = namedtuple('Surface', 'vox px')
FrameGT = namedtuple('FrameGT', 'lab dep K R t L cam_all ok_all')
_OFF = 1 << 20


def voxelize(p, v=VOX):
    """점 → 2cm 셀 중심 (중복 제거, x·y·z 사전식 정렬). score_geometry.voxelize / score_frontend.voxelize 와 같은 값.
    np.unique(axis=0) 대신 셀 번호를 int64 하나로 접어서 정렬한다 (|셀| < 2^20 = 20km, 같은 순서)."""
    p = np.asarray(p)
    if not len(p):
        return np.zeros((0, 3), np.float64)
    k = np.floor(p.astype(np.float64, copy=False) / v).astype(np.int64)
    if np.abs(k).max() >= _OFF:
        return (np.unique(k, axis=0) + 0.5) * v
    key = np.unique(((k[:, 0] + _OFF) << 42) | ((k[:, 1] + _OFF) << 21) | (k[:, 2] + _OFF))
    k = np.stack([(key >> 42) - _OFF, ((key >> 21) & ((1 << 21) - 1)) - _OFF, (key & ((1 << 21) - 1)) - _OFF], 1)
    return (k + 0.5) * v


def backproject(u, v, Z, K):
    """score_geometry.score_sequence 안의 backproject 와 같은 식."""
    return np.stack([(u - K[0, 2]) / K[0, 0] * Z, (v - K[1, 2]) / K[1, 1] * Z, Z], -1)


def labels_meta(labels_dir):
    """라벨 폴더의 meta.json → dict(labels_dir, version, meta_sha1). meta.json 이 없으면 version·sha1 = None."""
    labels_dir = Path(labels_dir)
    try:
        raw = (labels_dir / 'meta.json').read_bytes()
    except OSError:
        return dict(labels_dir=str(labels_dir), version=None, meta_sha1=None)
    try:
        ver = json.loads(raw).get('version')
    except ValueError:
        ver = None
    return dict(labels_dir=str(labels_dir), version=ver, meta_sha1=hashlib.sha1(raw).hexdigest())


class LabelSurface:
    def __init__(self, seq_dir, labels_dir, max_range=5.0, voxel=VOX, seq=None):
        if seq is None:
            from paths import WS  # noqa: F401  (meridian_benchmark 경로)
            from meridian_benchmark.uhumans2 import UH2Sequence
            seq = UH2Sequence(seq_dir)
        self.seq, self.labels_dir, self.max_range, self.voxel = seq, Path(labels_dir), float(max_range), float(voxel)
        self.K = seq.camera_info()['K'].copy(); self.K[0, 2] -= CROP
        self.vv, self.uu = np.mgrid[0:H:1, 0:W:1]
        self.poses = {}

    def prefetch_poses(self, frames):
        """frames 의 GT pose 를 한 번의 camera_poses 호출로 캐시 (값은 프레임마다 부른 것과 비트 단위로 같다 — 모듈 docstring)."""
        frames = [int(f) for f in frames if int(f) not in self.poses]
        if frames:
            t, q = self.seq.camera_poses(self.seq.stamps_ns[frames], cam='left_cam')
            for i, f in enumerate(frames):
                self.poses[f] = (t[i:i + 1], q[i:i + 1])
        return self

    def meta(self):
        return labels_meta(self.labels_dir)

    def frame_geometry(self, frame):
        """→ (라벨 480x640, depth 480x640 crop, K, R 3x3, t (3,)). 라벨 PNG 가 없으면 FileNotFoundError."""
        path = self.labels_dir / f'{int(frame):06d}.png'
        lab = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if lab is None:
            raise FileNotFoundError(f'GT 라벨 PNG 없음: {path} — gt_labels_2d.py 로 이 keyframe 라벨을 먼저 만들 것')
        dep = self.seq.depth(int(frame))[:, CROP:CROP + W]
        if int(frame) in self.poses:
            t, q = self.poses[int(frame)]
        else:
            t, q = self.seq.camera_poses(self.seq.stamps_ns[[int(frame)]], cam='left_cam')
        R = Rotation.from_quat(q[0]).as_matrix()
        return lab, dep, self.K, R, t[0]

    def frame(self, frame, stride=1):
        """라벨·depth·pose 를 읽고 stride 픽셀 간격으로 역투영 → FrameGT. score_geometry 의 옛 식:
        vv, uu = mgrid[0:480:stride, 0:640:stride] · L = lab[::stride, ::stride] · Z = dep[::stride, ::stride] ·
        cam_all = backproject(uu, vv, Z) · ok_all = (Z > 0) & (‖cam_all‖ ≤ max_range)."""
        lab, dep, K, R, t = self.frame_geometry(frame)
        if stride == 1:
            vv, uu, L, Z = self.vv, self.uu, lab, dep
        else:
            vv, uu = np.mgrid[0:H:stride, 0:W:stride]
            L, Z = lab[::stride, ::stride], dep[::stride, ::stride]
        cam_all = backproject(uu, vv, Z, K)
        ok_all = (Z > 0) & (np.linalg.norm(cam_all, axis=-1) <= self.max_range)
        return FrameGT(lab, dep, K, R, t, L, cam_all, ok_all)

    @staticmethod
    def raw(fg, ep, mask=None):
        """GT 원점 = cam_all[ok_all & (L == ep+1)] @ R.T + t (world, float64)."""
        m = fg.ok_all & (fg.L == int(ep) + 1) if mask is None else mask
        return fg.cam_all[m] @ fg.R.T + fg.t

    def at(self, frame):
        fg = self.frame(frame)
        out = {}
        vals = np.unique(fg.lab[fg.ok_all & (fg.lab > 0)])
        for val in vals:
            m = fg.ok_all & (fg.lab == val)
            out[int(val) - 1] = Surface(voxelize(self.raw(fg, int(val) - 1, m), self.voxel), int(m.sum()))
        return out
