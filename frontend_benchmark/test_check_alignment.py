#!/usr/bin/env python3
"""check_alignment.py 게이트 검증 (감사 C17) — 프레임 번호가 통째로 밀린 실행을 잡는가. 실제 데이터 읽기 전용.

감사 a02: frontend 점을 depth(f+d)+pose(f+d) 로 비교하면 (depth 와 pose 가 함께 밀림) d=±1·±2 에서도
p50 0.33cm · p90 0.79cm 로 옛 게이트(p99 ≤ 3cm)를 통과했다 — 정적 장면에서는 이웃 프레임도 같은 표면이기 때문.
여기서는 실제 apartment_s1_00h frontend_output.h5 에서 keyframe 30개를 골라 작은 h5 를 만들고
kf/frame_idx 만 s 칸 밀어(점은 그대로 — 'frame g 의 점을 frame g+s 라고 주장') check_alignment.py CLI 를 돌린다.
  s = 0      → 종료 코드 0 (통과)
  s = +1, −1 → 종료 코드 ≠ 0 (게이트 실패)
출력은 임시 폴더에만 쓴다. 실제 데이터는 paths.REAL_RUNS(FB_REAL_RUNS, FB_RUNS 와 무관 — 수정 ⑤ E4)에서 읽고,
없으면 'SKIP <이유>' 를 찍고 종료 0 (run_tests.py 가 skip 으로 보고한다).
실행: python test_check_alignment.py [시퀀스 (기본 apartment_s1_00h)]
"""
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths as P  # noqa: E402

FAILS = []


def check(name, got, want):
    ok = got == want
    print(f'  {"OK " if ok else "FAIL"} {name}: got={got!r} want={want!r}')
    if not ok:
        FAILS.append(name)


def small_run(src_h5, out_dir, shift, n_kf=30):
    """keyframe n_kf 개만 담은 frontend_output.h5 (점·관측 재색인), kf/frame_idx 에 shift 를 더한다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(src_h5, 'r') as f, h5py.File(out_dir / 'frontend_output.h5', 'w') as g:
        kf = f['kf/frame_idx'][:]; o0 = f['kf/obs_start'][:]; no = f['kf/n_obs'][:]
        ps = f['obs/points_start'][:]; pn = f['obs/points_num'][:]
        n_frames = int(f['frames/frame_idx'].shape[0]) if 'frames/frame_idx' in f else int(kf.max()) + 10
        ok = np.flatnonzero((kf >= 3) & (kf <= n_frames - 4) & (no > 0))
        pick = ok[np.linspace(0, len(ok) - 1, n_kf).astype(int)]
        pts, starts, nums, kobs0, knobs = [], [], [], [], []
        cur_p = cur_o = 0
        for k in pick:
            kobs0.append(cur_o); knobs.append(int(no[k]))
            for o in range(o0[k], o0[k] + no[k]):
                p = f['points'][ps[o]:ps[o] + pn[o]]
                pts.append(p); starts.append(cur_p); nums.append(len(p)); cur_p += len(p); cur_o += 1
        g['kf/frame_idx'] = kf[pick].astype(np.int64) + shift
        g['kf/obs_start'] = np.array(kobs0, np.int64); g['kf/n_obs'] = np.array(knobs, np.int64)
        g['obs/points_start'] = np.array(starts, np.int64); g['obs/points_num'] = np.array(nums, np.int64)
        g['points'] = np.concatenate(pts).astype(np.float32)
    return out_dir


def run_gate(run_dir, s):
    r = subprocess.run([sys.executable, str(HERE / 'check_alignment.py'), '--run', str(run_dir), '--seq', str(P.seq_dir(s)),
                        '--gt', str(P.gt_h5(s)), '--n-kf', '30'], capture_output=True, text=True, timeout=900)
    tail = [l for l in (r.stdout + r.stderr).splitlines() if 'Warning' not in l and 'warnings.warn' not in l][-4:]
    print('     ' + '\n     '.join(tail))
    return r.returncode


def test_shift(s):
    """C17 프레임 번호 통째 밀림 → 게이트 실패, 밀림 없음 → 통과"""
    src = P.REAL_RUNS / f'uHumans2_{s}' / 'frontend_output.h5'
    miss = [str(x) for x in (src, P.seq_dir(s), P.gt_h5(s)) if not Path(x).exists()]
    if miss:
        print(f'  SKIP {s}: 실제 데이터 없음 {miss} (FB_REAL_RUNS · FB_DATA · FB_GT 확인)'); return
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for shift, want_fail in ((0, False), (1, True), (-1, True)):
            rd = small_run(src, tmp / f'shift{shift:+d}', shift)
            rc = run_gate(rd, s)
            check(f'{s} 프레임 {shift:+d} 칸 밀림 → {"게이트 실패" if want_fail else "통과"}', rc != 0, want_fail)


def main():
    s = next((a for a in sys.argv[1:] if not a.startswith('-')), 'apartment_s1_00h')
    for fn in (test_shift,):
        print(f'[{fn.__name__}] {fn.__doc__.strip()}')
        try:
            fn(s)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc(limit=3)
            print(f'  FAIL {fn.__name__}: 예외 {type(e).__name__}: {e}')
            FAILS.append(fn.__name__)
    print('\n결과:', '전부 통과' if not FAILS else f'실패 {len(FAILS)}개 {FAILS}')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
