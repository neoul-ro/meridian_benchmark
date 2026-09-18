#!/usr/bin/env python3
"""frontend_provenance.py 검증 (수정 ⑤ 후속 F1) — frontend(GPU) 실행을 다시 할지 파일 시각이 아니라 출처로 정한다.

GPU 는 쓰지 않는다. 가짜 run_frontend.py · 가짜 src 트리 · 가짜 엔진 파일로 기록(record) → 확인(check) 을 되풀이한다.
  P1 기록 직후                                  → 최신 (종료 0)
  P2 run_frontend.py 의 docstring·주석만 바꿈     → 최신 (docstring 뺀 AST 가 같다 — 감사 C 의 ast_check 와 같은 기준)
  P3 run_frontend.py 로직 바꿈                   → 다시 필요 (종료 1, 이유에 run_frontend.py)
  P4 src/meridian_frontend 파일 바꿈              → 다시 필요 (이유에 frontend 소스) · __pycache__/*.pyc 만 생김 → 최신
  P5 엔진 파일 내용 바꿈 (크기 같음)               → 다시 필요 (이유에 엔진 이름)
  P6 데이터셋 시퀀스 경로 바뀜                     → 다시 필요
  P7 기록 없음 · frontend_output.h5 없음           → 다시 필요
  P8 backfill 기록은 backfill 표시를 남기고 확인은 최신
  P9 실제 6개 실행 결과: 기록(backfill 포함)이 지금 코드·src·엔진·데이터셋과 같다 (paths.REAL_RUNS, 없으면 SKIP)
실행: python test_frontend_provenance.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths as P  # noqa: E402

FAILS = []
PY = sys.executable

FAKE_RF = '''#!/usr/bin/env python3
"""가짜 frontend 러너 — 원래 설명."""
import sys

MODE = 1   # 로직 값


def main():
    """함수 설명."""
    return MODE


if __name__ == '__main__':
    sys.exit(main())
'''


def check(name, got, want):
    ok = got == want
    print(f'  {"OK " if ok else "FAIL"} {name}: got={got!r} want={want!r}')
    if not ok:
        FAILS.append(name)


def fixture(tmp):
    tmp = Path(tmp)
    code = tmp / 'run_frontend.py'; code.write_text(FAKE_RF)
    src = tmp / 'src' / 'meridian_frontend'; (src / 'meridian_frontend').mkdir(parents=True)
    (src / 'meridian_frontend' / 'frontend.py').write_text('def assemble():\n    return 1\n')
    msgs = tmp / 'src' / 'meridian_frontend_msgs'; (msgs / 'msg').mkdir(parents=True)
    (msgs / 'msg' / 'TrackletSet.msg').write_text('int32 id\n')
    eng = tmp / 'models'; eng.mkdir()
    (eng / 'fastsam.plan').write_bytes(b'A' * 64); (eng / 'clip_image.plan').write_bytes(b'B' * 64)
    seq = tmp / 'data' / 'uHumans2_x'; seq.mkdir(parents=True)
    run = tmp / 'runs' / 'uHumans2_x'; run.mkdir(parents=True)
    (run / 'frontend_output.h5').write_bytes(b'h5')
    (run / 'run_meta.json').write_text(json.dumps(dict(n_frames=10, built_at='2026-09-17 20:32:20'), indent=2))
    args = ['--run', str(run), '--seq', str(seq), '--code', str(code), '--src', str(src), str(msgs),
            '--engines', str(eng / 'fastsam.plan'), str(eng / 'clip_image.plan'), '--cache', str(tmp / 'sha1.json')]
    return dict(tmp=tmp, code=code, src=src, msgs=msgs, eng=eng, seq=seq, run=run, args=args)


def cli(cmd, args):
    r = subprocess.run([PY, str(HERE / 'frontend_provenance.py'), cmd, *args], capture_output=True, text=True, timeout=300)
    return r.returncode, (r.stdout + r.stderr).strip()


def later(p):
    t = time.time() + 5
    os.utime(p, (t, t))


def test_cases():
    """P1~P8 기록 → 확인"""
    with tempfile.TemporaryDirectory() as d:
        fx = fixture(d)
        rc, out = cli('record', fx['args'])
        check('P1 record 종료 0', rc, 0)
        meta = json.loads((fx['run'] / 'run_meta.json').read_text())
        check('P1 run_meta.json 기존 키 유지', meta.get('n_frames'), 10)
        pv = meta.get('provenance') or {}
        check('P1 기록 키', sorted(k for k in ('run_frontend_ast_sha1', 'frontend_src_sha1', 'engines', 'dataset_seq') if k in pv),
              ['dataset_seq', 'engines', 'frontend_src_sha1', 'run_frontend_ast_sha1'])
        rc, out = cli('check', fx['args']); print('     ' + out)
        check('P1 기록 직후 → 최신 (종료 0)', rc, 0)

        fx['code'].write_text(FAKE_RF.replace('원래 설명.', '설명을 고침 — 주석만.').replace('# 로직 값', '# 주석도 고침')
                              .replace('"""함수 설명."""', '"""함수 설명을 길게 고침."""'))
        later(fx['code'])
        rc, out = cli('check', fx['args']); print('     ' + out)
        check('P2 docstring·주석만 바뀜 (파일은 더 새로움) → 최신', rc, 0)

        fx['code'].write_text(FAKE_RF.replace('MODE = 1', 'MODE = 2'))
        rc, out = cli('check', fx['args']); print('     ' + out)
        check('P3 로직 바뀜 → 다시 필요 (종료 1)', rc, 1)
        check('P3 이유에 run_frontend.py', 'run_frontend.py' in out, True)
        fx['code'].write_text(FAKE_RF)

        f = fx['src'] / 'meridian_frontend' / 'frontend.py'
        (fx['src'] / 'meridian_frontend' / '__pycache__').mkdir()
        (fx['src'] / 'meridian_frontend' / '__pycache__' / 'frontend.cpython-310.pyc').write_bytes(b'cache')
        rc, out = cli('check', fx['args'])
        check('P4 __pycache__ 만 생김 → 최신', rc, 0)
        f.write_text('def assemble():\n    return 2\n')
        rc, out = cli('check', fx['args']); print('     ' + out)
        check('P4 frontend src 바뀜 → 다시 필요', rc, 1)
        check('P4 이유에 frontend 소스', 'frontend 소스' in out, True)
        f.write_text('def assemble():\n    return 1\n')
        (fx['msgs'] / 'msg' / 'TrackletSet.msg').write_text('int64 id\n')
        rc, out = cli('check', fx['args'])
        check('P4 msgs 바뀜 → 다시 필요', rc, 1)
        (fx['msgs'] / 'msg' / 'TrackletSet.msg').write_text('int32 id\n')
        rc, _ = cli('check', fx['args'])
        check('P4 되돌리면 최신', rc, 0)

        e = fx['eng'] / 'clip_image.plan'
        st = e.stat()
        e.write_bytes(b'C' * 64); os.utime(e, ns=(st.st_atime_ns, st.st_mtime_ns))   # 크기·mtime 같고 내용만 다름
        rc, out = cli('check', fx['args'] + ['--no-cache']); print('     ' + out)
        check('P5 엔진 내용 바뀜 → 다시 필요', rc, 1)
        check('P5 이유에 엔진 이름', 'clip_image.plan' in out, True)
        e.write_bytes(b'B' * 64)

        seq2 = fx['tmp'] / 'data' / 'uHumans2_y'; seq2.mkdir()
        args2 = list(fx['args']); args2[args2.index('--seq') + 1] = str(seq2)
        rc, out = cli('check', args2)
        check('P6 데이터셋 경로 바뀜 → 다시 필요', (rc, '데이터셋' in out), (1, True))

        meta = json.loads((fx['run'] / 'run_meta.json').read_text()); meta.pop('provenance')
        (fx['run'] / 'run_meta.json').write_text(json.dumps(meta))
        rc, out = cli('check', fx['args']); print('     ' + out)
        check('P7 기록 없음 → 다시 필요', (rc, '기록 없음' in out), (1, True))
        (fx['run'] / 'frontend_output.h5').unlink()
        rc, out = cli('check', fx['args'])
        check('P7 frontend_output.h5 없음 → 다시 필요', (rc, 'frontend_output.h5' in out), (1, True))
        (fx['run'] / 'frontend_output.h5').write_bytes(b'h5')

        note = fx['tmp'] / 'evidence.json'
        note.write_text(json.dumps(dict(note='기존 실행 — 다시 돌리지 않고 기록만 채움', evidence=['ast 같음'])))
        rc, out = cli('record', fx['args'] + ['--backfill', str(note)])
        pv = json.loads((fx['run'] / 'run_meta.json').read_text())['provenance']
        check('P8 backfill 표시', (pv.get('backfill') or {}).get('note'), '기존 실행 — 다시 돌리지 않고 기록만 채움')
        rc, _ = cli('check', fx['args'])
        check('P8 backfill 기록으로 최신', rc, 0)


def test_real_runs():
    """P9 실제 6개 실행 결과의 기록이 지금 입력과 같다 (backfill 표시 포함)"""
    runs = [P.REAL_RUNS / f'uHumans2_{s}' for s in P.SEQUENCES]
    miss = [str(r) for r in runs if not (r / 'frontend_output.h5').exists()]
    if miss or not P.FRONTEND_SRC.exists():
        print(f'  SKIP 실제 데이터 없음 (REAL_RUNS={P.REAL_RUNS}): {miss[:2]}')
        return
    with tempfile.TemporaryDirectory() as d:
        cache = Path(d) / 'sha1.json'
        real_cache = P.REAL_RUNS / '.cache' / 'sha1.json'
        if real_cache.exists():
            shutil.copy(real_cache, cache)                               # 큰 파일(엔진·src 200MB) 해시 재사용 — 크기·mtime 이 같을 때만 쓰인다
        for s, run in zip(P.SEQUENCES, runs):
            args = ['--run', str(run), '--seq', str(P.seq_dir(s)), '--code', str(HERE / 'run_frontend.py'),
                    '--src', str(P.FRONTEND_SRC), str(P.FRONTEND_MSGS),
                    '--engines', str(P.MODELS / 'frontend_rtx3060/fastsam.plan'), str(P.MODELS / 'frontend_rtx3060/clip_image.plan'),
                    '--cache', str(cache)]
            rc, out = cli('check', args)
            pv = json.loads((run / 'run_meta.json').read_text()).get('provenance') or {}
            check(f'{s}: 기록과 지금 입력 같음 → frontend 다시 안 돌림 ({out[:60]})', rc, 0)
            check(f'{s}: backfill 표시 있음', bool((pv.get('backfill') or {}).get('note')), True)


def main():
    for fn in (test_cases, test_real_runs):
        print(f'[{fn.__name__}] {fn.__doc__.strip()}')
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc(limit=3)
            print(f'  FAIL {fn.__name__}: 예외 {type(e).__name__}: {e}')
            FAILS.append(fn.__name__)
    print('\n결과:', '전부 통과' if not FAILS else f'실패 {len(FAILS)}개 {FAILS}')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
