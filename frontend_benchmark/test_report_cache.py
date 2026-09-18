#!/usr/bin/env python3
"""report.py 캐시 검증 (감사 C4 · C5 · C6 · C9 · C16) — 감사 a08 시나리오를 '내용이 실제로 바뀐' 경우로 재현.

a08 은 mtime(수정 시각)만 조작했다. 내용 기반 캐시에서는 mtime 만 바뀐 경우 다시 계산하지 않는 것이 맞으므로
(아래 R2b · R8b), 여기서는 내용을 실제로 바꿔 '다시 계산해야 하는데 안 하는' 결함을 확인한다.

트리 (임시 폴더, 실제 결과·저장소는 읽기만)
  code/   이 폴더(또는 인자로 준 폴더)의 *.py 복사본 — report 가 import 하는 채점기 코드
  bench/meridian_benchmark/   __init__.py · uhumans2.py 복사본 (FB_BENCH_PKG)
  gt/ · gt_alt/   GT h5 복사본 · 내용이 다른 GT (h5 속성 하나 추가, 옛 mtime)
  runs/   uHumans2_apartment_s1_00h: 작은 frontend_output.h5 (keyframe 3개) · run_meta.json · 채점 결과 '정답' 복사본
          gt_vis/<seq>.npz · gt_labels_2d/<seq>/difficulty.npz
  logs/   run_<seq>.log
채점 함수(S.score · S2 · SG · SM · GD.build)는 스텁: 호출을 기록하고 정답 결과 파일을 복사한다 (실제 채점기와 같은 파일을 씀).
GL.build(라벨)는 진짜 함수를 keyframe 3개에 돌린다.

시나리오 (각각 새 프로세스에서 report.main)
  R0  처음                                    → (기록만)
  R1  아무것도 안 바뀜                          → 아무 단계도 안 부름 · runs/ 아래 어떤 파일도 안 씀 (C16)
  R2  GT h5 내용 바뀜 (새 mtime)                → 3D · 라벨(3장 전부) · 난이도 · 2D · 추적 (C4)
  R2b GT h5 mtime 만 바뀜                       → 아무것도 안 함 (내용 같음)
  R3  FB_GT 를 내용 다른 GT 폴더로 (옛 mtime)    → R2 와 같음 (C4)
  R4  uhumans2.py 코드 변경                     → 3D · 라벨 · 난이도 · 기하 (그 모듈을 import 하는 단계, C4)
  R5  gt_labels_2d.py 코드 변경 (PNG 전부 있음)  → 라벨 3장 전부 다시 (C5)
  R6  gt_labels_2d.py 변경 + PNG 1장 지움       → 라벨 3장 전부 다시, 옛·새 섞임 없음 (C5)
  R7  score.json 이 다른 params(--dup-policy count --tau 0.1)로 덮임 → 3D 다시 · 머리말은 다시 채점한 params (C6)
  R8  gt_vis npz 를 내용 다른 옛 백업으로 (cp -p) → 3D · 추적 (C4)
  R8b gt_vis npz mtime 만 과거로                 → 아무것도 안 함
  R9  시퀀스 6개 중 1개만 있음                    → 종료 코드 ≠ 0 · summary.md 에 누락 5개 명시 (C9)
      --seqs apartment_s1_00h                   → 종료 코드 0 · 제목 '1개 시퀀스'
실제 데이터 경로 [수정 ⑤ E4]  paths.REAL_RUNS(FB_REAL_RUNS, 기본 <WS>/runs/frontend_eval) — FB_RUNS 와 무관.
  필요한 실제 입력(GT h5 · 데이터셋 · frontend_output.h5 · gt_vis · difficulty.npz · 채점 JSON 4개)이 없으면 'SKIP <이유>' 후 종료 0.
  score_kf_gt.csv 는 스텁이 복사만 하므로 실제 파일이 없으면(옛 형식 결과) 자리 채움 파일을 쓴다.
  3D 스텁은 실제 채점기처럼 score.json params 의 labels_dir · labels_version · labels_meta_sha1 을 지금 라벨 폴더 값으로 쓰고,
  추적 스텁은 그 값을 score.json 에서 옮긴다 (score_mot.score_sequence 와 같은 동작).
실행: python test_report_cache.py [코드 폴더 (기본: 이 파일 폴더)]
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import h5py

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths as REAL  # noqa: E402

FAILS = []
SEQ = 'apartment_s1_00h'; NAME = f'uHumans2_{SEQ}'
GOLD_FILES = ('score.json', 'score_episodes.csv', 'score_observations.csv', 'score_kf_gt.csv', 'score_2d.json', 'score_2d_gt.csv',
              'score_2d_pred.csv', 'score_geometry.json', 'score_geometry.csv', 'score_mot.json', 'score_mot_timeline.json')
PLACEHOLDER_OK = ('score_kf_gt.csv',)                 # 옛 형식 실제 결과에는 없음 — 스텁은 복사만 한다
STEP_FILES = dict(score3d=('score.json', 'score_episodes.csv', 'score_observations.csv', 'score_kf_gt.csv'),
                  seg2d=('score_2d.json', 'score_2d_gt.csv', 'score_2d_pred.csv'),
                  geometry=('score_geometry.json', 'score_geometry.csv'),
                  mot=('score_mot.json', 'score_mot_timeline.json'))


def check(name, got, want):
    ok = got == want
    print(f'  {"OK " if ok else "FAIL"} {name}: got={got!r} want={want!r}')
    if not ok:
        FAILS.append(name)


DRIVER = r'''
import json, os, shutil, sys
from pathlib import Path
SIM = Path(sys.argv[1]); args = json.loads(sys.argv[2])
sys.path.insert(0, str(SIM / 'code'))
import report as R
calls = []
GOLD = SIM / 'gold'
RUN = SIM / 'runs' / 'uHumans2_apartment_s1_00h'
FILES = json.loads(os.environ['STEP_FILES'])
def stub(name):
    def f(*a, **kw):
        calls.append(name)
        for fn in FILES[name]:
            shutil.copyfile(GOLD / fn, RUN / fn)
        j = json.loads((RUN / FILES[name][0]).read_text())
        if name == 'score3d' and kw.get('labels_dir') is not None:      # 실제 score_frontend 처럼 쓴 라벨의 출처를 기록
            import gt_surface as GS
            lm = GS.labels_meta(kw['labels_dir'])
            j.setdefault('params', {}).update(labels_dir=lm['labels_dir'], labels_version=lm['version'], labels_meta_sha1=lm['meta_sha1'])
            (RUN / 'score.json').write_text(json.dumps(j, indent=2, ensure_ascii=False))
        if name == 'mot':                                              # score_mot 은 score.json 의 라벨 출처를 옮겨 적는다
            sp = json.loads((RUN / 'score.json').read_text()).get('params') or {}
            j.setdefault('params', {}).update({k: sp.get(k) for k in ('labels_version', 'labels_meta_sha1')})
            (RUN / 'score_mot.json').write_text(json.dumps(j, indent=2, ensure_ascii=False))
        return j
    return f
R.S.score = stub('score3d'); R.S2.score_sequence = stub('seg2d'); R.SG.score_sequence = stub('geometry'); R.SM.score_sequence = stub('mot')
def gd(seq, gt, labels, frames, *a, **kw):
    calls.append('difficulty'); shutil.copyfile(GOLD / 'difficulty.npz', Path(labels) / 'difficulty.npz'); return Path(labels) / 'difficulty.npz'
R.GD.build = gd
real_gl = R.GL.build
def gl(*a, **kw):
    msgs = []
    kw['log'] = msgs.append
    out = real_gl(*a, **kw)
    m = [re.search(r'새로 만든 것 (\d+)', x) for x in msgs]
    n = next((int(x.group(1)) for x in m if x), -1)
    calls.append(f'labels:{n}')
    return out
import re
R.GL.build = gl
import meridian_benchmark.uhumans2 as U
sys.argv = ['report.py'] + args
try:
    rc = R.main()
except SystemExit as e:
    rc = e.code
rc = 0 if rc is None else (rc if isinstance(rc, int) else 1)
md = Path(os.environ['FB_RUNS']) / 'summary.md'
lines = md.read_text().splitlines() if md.exists() else []
print('DRIVER_RESULT ' + json.dumps(dict(calls=calls, rc=rc, md=lines, uhumans2=U.__file__), ensure_ascii=False))
'''


def real_missing():
    """실제 데이터(FB_RUNS 와 무관한 paths.REAL_RUNS 등) 중 빠진 것 → 이유 목록."""
    run = REAL.REAL_RUNS / NAME
    need = [REAL.gt_h5(SEQ), REAL.seq_dir(SEQ), REAL.BENCH_PKG / 'meridian_benchmark' / 'uhumans2.py', run / 'frontend_output.h5',
            run / 'run_meta.json', REAL.REAL_RUNS / 'gt_vis' / f'{NAME}.npz', REAL.REAL_RUNS / 'gt_labels_2d' / NAME / 'difficulty.npz']
    need += [run / f for f in GOLD_FILES if f not in PLACEHOLDER_OK]
    return [str(p) for p in need if not Path(p).exists()]


def build_sim(sim, code_dir):
    code = sim / 'code'
    shutil.copytree(code_dir, code, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    bench = sim / 'bench' / 'meridian_benchmark'; bench.mkdir(parents=True)
    src_pkg = REAL.BENCH_PKG / 'meridian_benchmark'
    shutil.copy(src_pkg / '__init__.py', bench / '__init__.py'); shutil.copy(src_pkg / 'uhumans2.py', bench / 'uhumans2.py')
    real_run = REAL.REAL_RUNS / NAME
    gold = sim / 'gold'; gold.mkdir()
    for f in GOLD_FILES:
        if (real_run / f).exists():
            shutil.copy(real_run / f, gold / f)
        else:
            (gold / f).write_text(f'placeholder,{f}\n')              # PLACEHOLDER_OK 만 (real_missing 이 나머지를 막는다)
    shutil.copy(REAL.REAL_RUNS / 'gt_labels_2d' / NAME / 'difficulty.npz', gold / 'difficulty.npz')
    run = sim / 'runs' / NAME; run.mkdir(parents=True)
    shutil.copy(real_run / 'run_meta.json', run / 'run_meta.json')
    with h5py.File(real_run / 'frontend_output.h5', 'r') as s, h5py.File(run / 'frontend_output.h5', 'w') as d:
        d['kf/frame_idx'] = s['kf/frame_idx'][:3]
        d['obs/mask_bits'] = s['obs/mask_bits'][:2]
    (sim / 'runs' / 'gt_vis').mkdir()
    shutil.copy(REAL.REAL_RUNS / 'gt_vis' / f'{NAME}.npz', sim / 'runs' / 'gt_vis' / f'{NAME}.npz')
    (sim / 'gt').mkdir(); shutil.copy(REAL.gt_h5(SEQ), sim / 'gt' / f'{NAME}.h5')
    (sim / 'logs').mkdir()
    log = REAL.LOGS / f'run_{SEQ}.log'
    (sim / 'logs' / f'run_{SEQ}.log').write_text(log.read_text() if log.exists() else '')
    old = time.time() - 3 * 3600
    for p in list(code.glob('*.py')) + [bench / 'uhumans2.py', bench / '__init__.py', run / 'frontend_output.h5', run / 'run_meta.json',
                                        sim / 'runs' / 'gt_vis' / f'{NAME}.npz', sim / 'gt' / f'{NAME}.h5']:
        os.utime(p, (old, old))
    alt = sim / 'gt_alt'; alt.mkdir(); shutil.copy(sim / 'gt' / f'{NAME}.h5', alt / f'{NAME}.h5')
    with h5py.File(alt / f'{NAME}.h5', 'r+') as f:
        f.attrs['audit_marker'] = 'DIFFERENT GT'
    os.utime(alt / f'{NAME}.h5', (old - 7200, old - 7200))


def run_report(sim, args=(), gt_dir=None):
    env = dict(os.environ, MERIDIAN_WS=str(sim), FB_DATA=str(REAL.DATA), FB_GT=str(gt_dir or sim / 'gt'),
               FB_BENCH_PKG=str(sim / 'bench'), FB_RUNS=str(sim / 'runs'), FB_LOGS=str(sim / 'logs'),
               STEP_FILES=json.dumps(STEP_FILES))
    (sim / 'driver.py').write_text(DRIVER)
    t0 = time.time()
    r = subprocess.run([sys.executable, str(sim / 'driver.py'), str(sim), json.dumps(list(args))], env=env, capture_output=True,
                       text=True, timeout=900)
    line = next((l for l in r.stdout.splitlines() if l.startswith('DRIVER_RESULT ')), None)
    if line is None:
        print(r.stdout[-1500:], r.stderr[-2500:])
        raise RuntimeError('driver 실패')
    d = json.loads(line[len('DRIVER_RESULT '):])
    d['seconds'] = round(time.time() - t0, 1)
    return d


def snapshot(root):
    return {str(p.relative_to(root)): (p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(root.rglob('*')) if p.is_file()}


def steps(d):
    out = set()
    for c in d['calls']:
        out.add('labels' if c.startswith('labels:') and c != 'labels:0' else ('' if c == 'labels:0' else c))
    out.discard('')
    return out


def label_n(d):
    return max([int(c.split(':')[1]) for c in d['calls'] if c.startswith('labels:')] or [0])


def header(d):
    """표 밖의 설명 줄 — 3D 매칭 params 는 M6 이후 맨 뒤 '## 8.' 절에 있다."""
    return '\n'.join(l for l in d['md'] if not l.startswith('|') and not l.startswith('#'))[-3000:]


RESULTS = []


def scenario(name, d, want_steps=None, exact=False):
    got = steps(d)
    RESULTS.append(dict(scenario=name, recomputed=sorted(got), labels_rebuilt=label_n(d), rc=d['rc'], seconds=d['seconds']))
    print(f'  · {name}: 다시 계산 {sorted(got)} · 라벨 새로 {label_n(d)}장 · 종료 {d["rc"]} · {d["seconds"]}s')
    if want_steps is not None:
        if exact:
            check(f'{name}: 다시 계산 = {sorted(want_steps)}', sorted(got), sorted(want_steps))
        else:
            check(f'{name}: 다시 계산 ⊇ {sorted(want_steps)} (빠진 단계 목록이 비어야 함)', sorted(set(want_steps) - got), [])


def test_cache(code_dir):
    """C4·C5·C6·C9·C16 캐시 시나리오 R0~R9"""
    miss = real_missing()
    if miss:
        print(f'  SKIP 실제 데이터 없음 (paths.REAL_RUNS={REAL.REAL_RUNS}, FB_REAL_RUNS 로 바꿈): {miss[:4]}')
        return
    with tempfile.TemporaryDirectory() as dd:
        sim = Path(dd)
        build_sim(sim, code_dir)
        labels = sim / 'runs' / 'gt_labels_2d' / NAME
        d = run_report(sim, ['--seqs', SEQ]); scenario('R0 처음', d)
        print('     uhumans2 =', d['uhumans2'])
        check('R0 종료 코드 0', d['rc'], 0)

        before = snapshot(sim / 'runs')
        d = run_report(sim, ['--seqs', SEQ]); scenario('R1 변경 없음', d, set(), exact=True)
        after = snapshot(sim / 'runs')
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        check('R1 runs/ 아래 다시 쓴 파일 없음 (meta.json·캐시·summary 포함)', changed, [])
        RESULTS[-1]['files_rewritten'] = changed

        gt = sim / 'gt' / f'{NAME}.h5'
        with h5py.File(gt, 'r+') as f:
            f.attrs['audit_marker_r2'] = 'CHANGED'
        d = run_report(sim, ['--seqs', SEQ]); scenario('R2 GT h5 내용 바뀜', d, {'score3d', 'labels', 'difficulty', 'seg2d', 'mot'})
        check('R2 라벨 3장 전부 다시', label_n(d), 3)

        t = time.time() + 5; os.utime(gt, (t, t))
        d = run_report(sim, ['--seqs', SEQ]); scenario('R2b GT h5 mtime 만 바뀜', d, set(), exact=True)

        d = run_report(sim, ['--seqs', SEQ], gt_dir=sim / 'gt_alt'); scenario('R3 FB_GT → 내용 다른 GT 폴더(옛 mtime)', d,
                                                                             {'score3d', 'labels', 'difficulty', 'seg2d', 'mot'})
        d = run_report(sim, ['--seqs', SEQ], gt_dir=sim / 'gt_alt'); scenario('R3 확인: 같은 GT 로 다시 → 없음', d, set(), exact=True)

        u = sim / 'bench' / 'meridian_benchmark' / 'uhumans2.py'
        u.write_text(u.read_text() + '\nAUDIT_R4_MARKER = 1\n')
        d = run_report(sim, ['--seqs', SEQ], gt_dir=sim / 'gt_alt'); scenario('R4 uhumans2.py 코드 변경', d,
                                                                             {'score3d', 'labels', 'difficulty', 'geometry'})

        g = sim / 'code' / 'gt_labels_2d.py'
        g.write_text(g.read_text() + '\nAUDIT_R5_MARKER = 1\n')
        d = run_report(sim, ['--seqs', SEQ], gt_dir=sim / 'gt_alt'); scenario('R5 gt_labels_2d.py 코드 변경', d, {'labels'})
        check('R5 라벨 3장 전부 다시 (새로 만든 것 3)', label_n(d), 3)

        g.write_text(g.read_text() + '\nAUDIT_R6_MARKER = 2\n')
        pngs = sorted(labels.glob('*.png'))
        pngs[1].unlink()
        t6 = time.time()
        d = run_report(sim, ['--seqs', SEQ], gt_dir=sim / 'gt_alt'); scenario('R6 코드 변경 + PNG 1장 삭제', d, {'labels'})
        check('R6 라벨 3장 전부 다시', label_n(d), 3)
        check('R6 옛 PNG 섞임 없음 (전부 이번 실행에 씀)', all(p.stat().st_mtime >= t6 - 1 for p in labels.glob('*.png')), True)

        sj = sim / 'runs' / NAME / 'score.json'
        j = json.loads(sj.read_text()); j['params']['dup_policy'] = 'count'; j['params']['tau_m'] = 0.1
        sj.write_text(json.dumps(j, indent=2, ensure_ascii=False))
        d = run_report(sim, ['--seqs', SEQ], gt_dir=sim / 'gt_alt'); scenario('R7 score.json 이 다른 params 로 덮임', d, {'score3d'})
        pj = json.loads(sj.read_text())['params']
        check('R7 다시 채점 후 params = 기본값 (hungarian, 0.2)', (pj.get('dup_policy'), pj.get('tau_m')), ('hungarian', 0.2))
        h = header(d)
        check('R7 머리말에 실제 params (dup_policy hungarian)', 'hungarian' in h, True)
        check('R7 머리말에 count 없음', 'count' in h, False)

        vis = sim / 'runs' / 'gt_vis' / f'{NAME}.npz'
        import numpy as np
        z = dict(np.load(vis)); keep = np.arange(len(z['ep_index'])) % 7 != 0
        old_vis = sim / 'vis_backup.npz'
        np.savez_compressed(old_vis, **{k: (v[keep] if getattr(v, 'shape', ()) and len(v) == len(keep) else v) for k, v in z.items()})
        shutil.copy2(old_vis, vis); tt = time.time() - 30 * 3600; os.utime(vis, (tt, tt))
        d = run_report(sim, ['--seqs', SEQ], gt_dir=sim / 'gt_alt'); scenario('R8 gt_vis npz → 내용 다른 옛 백업(cp -p)', d, {'score3d', 'mot'})

        tt = time.time() - 40 * 3600; os.utime(vis, (tt, tt))
        d = run_report(sim, ['--seqs', SEQ], gt_dir=sim / 'gt_alt'); scenario('R8b gt_vis mtime 만 과거로', d, set(), exact=True)

        d = run_report(sim, [], gt_dir=sim / 'gt_alt'); scenario('R9 시퀀스 6개 요청 · 1개만 있음', d)
        check('R9 종료 코드 ≠ 0', d['rc'] != 0, True)
        md = '\n'.join(d['md'])
        missing = [s for s in REAL.SEQUENCES if s != SEQ]
        check('R9 summary.md 에 누락 5개 이름', all(s in md for s in missing), True)
        check('R9 summary.md 에 "누락" 표시', '누락' in md, True)
        check('R9 제목이 6개가 아님', any(l.startswith('# ') and '6개 시퀀스' in l for l in d['md']), False)
        d = run_report(sim, ['--seqs', SEQ], gt_dir=sim / 'gt_alt'); scenario('R9 --seqs 한 개', d)
        check('R9 --seqs: 종료 코드 0', d['rc'], 0)
        check('R9 --seqs: 제목 1개 시퀀스', any(l.startswith('# ') and '1개 시퀀스' in l for l in d['md']), True)


def main():
    code_dir = Path(next((a for a in sys.argv[1:] if not a.startswith('-')), HERE)).resolve()
    for fn in (test_cache,):
        print(f'[{fn.__name__}] {fn.__doc__.strip()}')
        try:
            fn(code_dir)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc(limit=4)
            print(f'  FAIL {fn.__name__}: 예외 {type(e).__name__}: {e}')
            FAILS.append(fn.__name__)
    out = os.environ.get('CACHE_RESULTS_JSON')
    if out:
        Path(out).write_text(json.dumps(RESULTS, indent=1, ensure_ascii=False))
    print('\n결과:', '전부 통과' if not FAILS else f'실패 {len(FAILS)}개 {FAILS}')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
