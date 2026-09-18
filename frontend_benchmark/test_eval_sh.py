#!/usr/bin/env python3
"""eval.sh 검증 (감사 C7 · C11) — 가짜 스크립트·임시 폴더만 쓴다. GPU·실제 결과를 건드리지 않는다.

C7  산출물이 입력보다 오래됐으면 다시 만든다 (-nt), 정렬 게이트는 지금 frontend_output.h5 의 결과만 읽는다.
    eval.sh 의 함수 정의만 뽑아(sed 와 같은 방식, 감사 a10) bash 에서 run_seq 를 부른다. HERE 는 가짜 스크립트 폴더:
      run_frontend.py      frontend_output.h5 를 쓴다 (불리면 안 되는 상황만 만든다)
      check_alignment.py   정렬이 '깨진' 결과(p99 25cm, gate.pass=false)를 쓰고 종료 1
      gt_visibility.py     npz 를 쓴다
    사례 1 (감사 a10): 좋은 옛 alignment_check.json(과거 mtime) + 새로 생긴 frontend_output.h5
          → 정렬 검증을 다시 돌려 깨진 결과를 읽고 실패해야 한다 (옛 코드: 건너뛰고 옛 결과로 통과)
    사례 2: alignment_check.json 이 mtime 은 더 새롭지만 기록된 h5(크기·sha1)가 지금 h5 와 다름 (cp -p 로 옛 h5 복원 등)
          → 게이트 실패 (옛 코드: p99 만 보고 통과)
    사례 3: 전부 최신 · 기록된 h5 가 지금 h5 와 같음 · 게이트 통과 → 성공 (정상 경로 확인)
C11 eval.sh test
    FB_TEST_DIR 의 test_*.py 를 전부 돌리고, 원문 로그를 남기고, 실패하면 traceback 끝부분을 보여 주고,
    RUNS/tests_summary.json (파일별 통과·OK 수) 을 쓴다. 예외로 죽는 테스트가 있으면 종료 코드 ≠ 0.
    run_tests.py --list: frontend_benchmark 의 test_*.py 전부(test_trackeval_parity · test_gt_labels_2d · test_gt_difficulty_2d 포함).
E4  테스트가 FB_RUNS(출력·샌드박스 폴더)에 기대지 않는다
    · paths.REAL_RUNS(실제 데이터, FB_REAL_RUNS) · paths.TRACKEVAL_PATH(TRACKEVAL_PATH) 기본값이 FB_RUNS 와 무관
    · 실제 데이터가 필요한 테스트 3개(report_cache · check_alignment · trackeval_parity)는 실제 데이터가 없으면
      'SKIP <이유>' 를 찍고 종료 0 → run_tests.py 가 skip 으로 보고 (pass · fail · 예외 아님)
    · run_tests.py 는 파일마다 SKIP 줄 수(n_skip)와 이유를 요약에 남긴다 — OK 와 SKIP 이 섞인 파일은 pass + n_skip
    · test_score_2d.py 를 직접 실행해도 라벨·난이도 테스트를 다시 돌리지 않는다 (예전 main 은 두 파일의 run() 까지 불렀다)
F1  frontend(GPU) 단계는 파일 시각(-nt)이 아니라 출처(run_meta.json provenance)로 다시 할지 정하고, GPU 단계는 명시적 허락이 있어야 돈다.
    GPU 를 쓰지 않는다 — 가짜 run_frontend.py 는 불리면 frontend_output.h5 에 RERUN-FRONTEND 를 쓴다.
      F1a run_frontend.py 의 docstring 만 바뀜(파일은 더 새로움)      → run_seq 가 frontend 를 다시 안 돌림 · 종료 0
      F1b 로직 바뀜 · 허락 없음(GPU≠1, 터미널 아님)                   → 종료 3 · h5 그대로 · 안내에 'eval.sh score' 와 'GPU=1'
      F1c frontend src 바뀜 · 허락 없음                               → 종료 3 · h5 그대로
      F1d 로직 바뀜 · GPU=1                                          → frontend 실행 · 출처 기록 갱신 (다시 확인하면 최신)
      F1e DRY_RUN=1 cmd_all                                          → 계획만 보이고 종료 0 · 아무것도 실행 안 함
      F1f cmd_all · 다시 필요한 시퀀스 있음 · 허락 없음              → 어떤 단계도 돌기 전에 종료 3 (정렬 검증 · 가시성 그대로)
      F1g FORCE=1 · 허락 없음                                        → 종료 3 (FORCE 만으로 GPU 를 쓰지 않는다)
"""
import hashlib
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

HERE = Path(__file__).resolve().parent
FAILS = []
PY = sys.executable


def check(name, got, want):
    ok = got == want
    print(f'  {"OK " if ok else "FAIL"} {name}: got={got!r} want={want!r}')
    if not ok:
        FAILS.append(name)


def functions_of(sh):
    """eval.sh 에서 함수 정의 블록만 (name() { … } 줄 단위). case 분기·환경 설정은 뺀다."""
    out, cur = [], None
    for line in sh.splitlines():
        if cur is None and re.match(r'^[A-Za-z_][A-Za-z0-9_]*\(\)\s*\{', line):
            cur = [line]
            if line.rstrip().endswith('}') and line.count('{') == line.count('}'):
                out.append(line); cur = None
            continue
        if cur is not None:
            cur.append(line)
            if line.startswith('}'):
                out.append('\n'.join(cur)); cur = None
    return '\n\n'.join(out)


FAKE_CHECK = r'''
import json, sys
a = sys.argv
run = a[a.index('--run') + 1]
json.dump({"input_alignment_cm": {"p50": 12.0, "p99": 25.0}, "gate": {"pass": False}, "marker": "RERUN"}, open(run + "/alignment_check.json", "w"))
sys.exit(1)
'''
FAKE_FRONTEND = r'''
"""가짜 frontend 러너 (eval.sh 검증용) — 설명."""
import sys

MODE = 1


def main():
    a = sys.argv
    open(a[a.index('--out') + 1] + "/frontend_output.h5", "w").write("RERUN-FRONTEND")


main()
'''
FAKE_VIS = r'''
import sys
a = sys.argv
open(a[a.index('--out') + 1], "w").write("vis")
'''


def sha1(p):
    return hashlib.sha1(Path(p).read_bytes()).hexdigest()


def good_json(h5):
    st = os.stat(h5)
    return {"input_alignment_cm": {"p50": 0.3, "p99": 0.8},
            "frame_shift": {"-1": {"within_1mm": 0.05}, "0": {"within_1mm": 0.25}, "1": {"within_1mm": 0.05}},
            "gate": {"input_p99_ok": True, "frame_shift_ok": True, "pass": True},
            "run_h5": {"path": str(h5), "size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha1": sha1(h5)}}


def setup(tmp, code_dir):
    fake = tmp / 'here'; fake.mkdir()
    for f in code_dir.glob('*.py'):                          # 진짜 모듈(게이트가 쓰면)은 그대로, 세 스크립트만 가짜
        shutil.copy(f, fake / f.name)
    (fake / 'check_alignment.py').write_text(FAKE_CHECK)
    (fake / 'run_frontend.py').write_text(FAKE_FRONTEND)
    (fake / 'gt_visibility.py').write_text(FAKE_VIS)
    s = 'apartment_s1_00h'
    ws = tmp / 'ws'; out = ws / 'runs'; run = out / f'uHumans2_{s}'; run.mkdir(parents=True); (out / 'gt_vis').mkdir()
    (ws / 'data' / f'uHumans2_{s}').mkdir(parents=True); (ws / 'gt').mkdir(); (ws / 'logs').mkdir(); (ws / 'models').mkdir()
    (ws / 'gt' / f'uHumans2_{s}.h5').write_text('gt')
    eng = ws / 'models' / 'frontend_rtx3060'; eng.mkdir()
    (eng / 'fastsam.plan').write_bytes(b'seg'); (eng / 'clip_image.plan').write_bytes(b'clip')
    fe = ws / 'src' / 'meridian_frontend' / 'meridian_frontend'; fe.mkdir(parents=True); (fe / 'frontend.py').write_text('X = 1\n')
    fm = ws / 'src' / 'meridian_frontend_msgs' / 'msg'; fm.mkdir(parents=True); (fm / 'T.msg').write_text('int32 a\n')
    return fake, ws, s


def prov(fake, ws, s, cmd='record'):
    """frontend_provenance.py (진짜 모듈 사본) 로 기록/확인 — eval.sh 가 넘기는 인자와 같게."""
    eng = ws / 'models' / 'frontend_rtx3060'
    r = subprocess.run([PY, str(fake / 'frontend_provenance.py'), cmd, '--run', str(ws / 'runs' / f'uHumans2_{s}'),
                        '--seq', str(ws / 'data' / f'uHumans2_{s}'), '--code', str(fake / 'run_frontend.py'),
                        '--src', str(ws / 'src' / 'meridian_frontend'), str(ws / 'src' / 'meridian_frontend_msgs'),
                        '--engines', str(eng / 'fastsam.plan'), str(eng / 'clip_image.plan'),
                        '--cache', str(ws / 'runs' / '.cache' / 'sha1.json')], capture_output=True, text=True, timeout=120)
    return r.returncode, (r.stdout + r.stderr).strip()


def age(p, sec):
    t = time.time() - sec
    os.utime(p, (t, t))


def run_seq(code_dir, fake, ws, s, call=None, force=0, gpu=0, dry=0):
    fns = functions_of((code_dir / 'eval.sh').read_text())
    script = f'''
set -euo pipefail
HERE={fake}; WS={ws}; OUT={ws}/runs; RUNS={ws}/runs; LOG={ws}/logs; LOGS={ws}/logs; GTDIR={ws}/gt; GT_DIR={ws}/gt
DATA={ws}/data; MODELS={ws}/models; PY={PY}; FORCE={force}; GPU={gpu}; DRY_RUN={dry}; GPU_APPROVED=0
FRONTEND_SRC={ws}/src/meridian_frontend; FRONTEND_MSGS={ws}/src/meridian_frontend_msgs
ALL_SEQS=({s})
s={s}   # 실제 eval.sh 는 run_seq 를 부르기 전에 전역 s 가 있다 (for s in … / s=$(resolve …))
{fns}
{call or f'run_seq {s}'}
'''
    r = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL)
    return r.returncode, r.stdout + r.stderr


def test_c7_stale_alignment(code_dir):
    """C7 사례 1 — 새 frontend 출력 뒤 옛 정렬 결과로 통과하지 않는다"""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); fake, ws, s = setup(tmp, code_dir)
        run = ws / 'runs' / f'uHumans2_{s}'
        h5 = run / 'frontend_output.h5'; h5.write_text('old-run'); prov(fake, ws, s)
        aj = run / 'alignment_check.json'; aj.write_text(json.dumps(good_json(h5)))
        for p in list(fake.glob('*.py')) + [ws / 'gt' / f'uHumans2_{s}.h5']:
            age(p, 5 * 86400)
        age(aj, 2 * 86400)
        h5.write_text('NEW-run-output')                            # frontend 를 다시 돌려 새 출력 (json 보다 새로움)
        age(h5, 3600)
        vis = ws / 'runs' / 'gt_vis' / f'uHumans2_{s}.npz'; vis.write_text('vis')
        rc, log = run_seq(code_dir, fake, ws, s)
        print('     ' + '\n     '.join(log.strip().splitlines()[-6:]))
        check('종료 코드 ≠ 0 (깨진 정렬로 실패)', rc != 0, True)
        check('alignment_check.json 을 다시 만듦 (RERUN)', 'RERUN' in aj.read_text(), True)
        check('frontend 는 다시 돌리지 않음 (h5 가 입력보다 새로움)', h5.read_text(), 'NEW-run-output')


def test_c7_gate_identity(code_dir):
    """C7 사례 2 — json 이 mtime 으로는 새롭지만 다른 h5 의 결과면 게이트 실패"""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); fake, ws, s = setup(tmp, code_dir)
        run = ws / 'runs' / f'uHumans2_{s}'
        h5 = run / 'frontend_output.h5'; h5.write_text('run-A'); prov(fake, ws, s)
        good = good_json(h5)
        h5.write_text('run-B-different-content!')                  # 다른 실행 결과로 바뀜 (cp -p 로 옛 mtime)
        for p in list(fake.glob('*.py')) + [ws / 'gt' / f'uHumans2_{s}.h5']:
            age(p, 5 * 86400)
        age(h5, 4 * 86400)
        aj = run / 'alignment_check.json'; aj.write_text(json.dumps(good)); age(aj, 3600)
        vis = ws / 'runs' / 'gt_vis' / f'uHumans2_{s}.npz'; vis.write_text('vis')
        rc, log = run_seq(code_dir, fake, ws, s)
        print('     ' + '\n     '.join(log.strip().splitlines()[-4:]))
        check('종료 코드 ≠ 0 (다른 h5 의 정렬 결과)', rc != 0, True)


def test_c7_fresh_ok(code_dir):
    """C7 사례 3 — 전부 최신이면 통과 (정상 경로)"""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); fake, ws, s = setup(tmp, code_dir)
        run = ws / 'runs' / f'uHumans2_{s}'
        h5 = run / 'frontend_output.h5'; h5.write_text('run-A'); prov(fake, ws, s)
        for p in list(fake.glob('*.py')) + [ws / 'gt' / f'uHumans2_{s}.h5']:
            age(p, 5 * 86400)
        age(h5, 4 * 86400)
        aj = run / 'alignment_check.json'; aj.write_text(json.dumps(good_json(h5))); age(aj, 3 * 86400)
        vis = ws / 'runs' / 'gt_vis' / f'uHumans2_{s}.npz'; vis.write_text('vis'); age(vis, 2 * 86400)
        rc, log = run_seq(code_dir, fake, ws, s)
        print('     ' + '\n     '.join(log.strip().splitlines()[-4:]))
        check('종료 코드 0', rc, 0)
        check('정렬 검증 다시 안 함', 'RERUN' in aj.read_text(), False)
        check('가시성 다시 안 함', vis.read_text(), 'vis')


def fresh_fixture(tmp, code_dir):
    """기록된 frontend 출력 · 통과한 정렬 결과 · 가시성까지 모두 최신인 시퀀스 하나 (파일 시각은 과거)."""
    tmp.mkdir(parents=True, exist_ok=True)
    fake, ws, s = setup(tmp, code_dir)
    run = ws / 'runs' / f'uHumans2_{s}'
    h5 = run / 'frontend_output.h5'; h5.write_text('run-A')
    rc, out = prov(fake, ws, s)
    assert rc == 0, out
    for p in list(fake.glob('*.py')) + [ws / 'gt' / f'uHumans2_{s}.h5']:
        age(p, 5 * 86400)
    age(h5, 4 * 86400)
    aj = run / 'alignment_check.json'; aj.write_text(json.dumps(good_json(h5))); age(aj, 3 * 86400)
    vis = ws / 'runs' / 'gt_vis' / f'uHumans2_{s}.npz'; vis.write_text('vis'); age(vis, 2 * 86400)
    return fake, ws, s, h5, aj, vis


def test_f1_frontend_provenance(code_dir):
    """F1 frontend 단계 = 출처 기준 · GPU 는 명시적 허락이 있어야 (가짜 러너, GPU 없음)"""
    with tempfile.TemporaryDirectory() as d:
        fake, ws, s, h5, aj, vis = fresh_fixture(Path(d) / 'a', code_dir)
        rf = fake / 'run_frontend.py'
        rf.write_text(rf.read_text().replace('설명.', '설명을 고쳤다 (docstring 만).'))            # 파일 시각이 h5 보다 새로움
        rc, log = run_seq(code_dir, fake, ws, s)
        print('     F1a ' + ' / '.join(log.strip().splitlines()[-3:]))
        check('F1a docstring 만 바뀜 → 종료 0', rc, 0)
        check('F1a frontend 다시 안 돌림 (h5 그대로)', h5.read_text(), 'run-A')

        rf.write_text(rf.read_text().replace('MODE = 1', 'MODE = 2'))
        rc, log = run_seq(code_dir, fake, ws, s)
        print('     F1b ' + ' / '.join(log.strip().splitlines()[-4:]))
        check('F1b 로직 바뀜 · 허락 없음 → 종료 3', rc, 3)
        check('F1b h5 그대로', h5.read_text(), 'run-A')
        check('F1b 안내에 채점만 하는 명령 (eval.sh score)', 'eval.sh score' in log, True)
        check('F1b 안내에 GPU=1', 'GPU=1' in log, True)
        check('F1b 이유에 run_frontend.py', 'run_frontend.py' in log, True)

        rc, log = run_seq(code_dir, fake, ws, s, gpu=1)
        check('F1d GPU=1 → frontend 실행 (h5 새로 씀)', h5.read_text(), 'RERUN-FRONTEND')
        rc2, out = prov(fake, ws, s, 'check')
        check('F1d 실행 뒤 출처 기록 갱신 → 확인하면 최신', rc2, 0)

    with tempfile.TemporaryDirectory() as d:
        fake, ws, s, h5, aj, vis = fresh_fixture(Path(d) / 'c', code_dir)
        (ws / 'src' / 'meridian_frontend' / 'meridian_frontend' / 'frontend.py').write_text('X = 2\n')
        rc, log = run_seq(code_dir, fake, ws, s)
        check('F1c frontend src 바뀜 · 허락 없음 → 종료 3 · h5 그대로', (rc, h5.read_text()), (3, 'run-A'))
        aj_before, vis_before = aj.read_text(), vis.read_text()
        rc, log = run_seq(code_dir, fake, ws, s, call='cmd_all', dry=1)
        print('     F1e ' + '\n         '.join(log.strip().splitlines()[-7:]))
        check('F1e DRY_RUN=1 → 종료 0', rc, 0)
        check('F1e 계획에 그 시퀀스와 이유 (화면 이름은 짧은 쪽, 재리뷰 N8)',
              (s.replace('_s1_', '_') in log) and 'frontend 소스' in log, True)
        check('F1e 아무것도 실행 안 함 (h5 · 정렬 · 가시성 그대로)', (h5.read_text(), aj.read_text() == aj_before, vis.read_text() == vis_before),
              ('run-A', True, True))
        age(vis, 10 * 86400)                                                       # 가시성이 오래돼 CPU 단계도 다시 할 상황
        rc, log = run_seq(code_dir, fake, ws, s, call='cmd_all')
        check('F1f cmd_all · 허락 없음 → 어떤 단계도 돌기 전에 종료 3', (rc, h5.read_text(), vis.read_text()), (3, 'run-A', 'vis'))

    with tempfile.TemporaryDirectory() as d:
        fake, ws, s, h5, aj, vis = fresh_fixture(Path(d) / 'g', code_dir)
        rc, log = run_seq(code_dir, fake, ws, s, force=1)
        check('F1g FORCE=1 · 허락 없음 → 종료 3 · h5 그대로', (rc, h5.read_text()), (3, 'run-A'))


def test_c11_eval_test(code_dir):
    """C11 eval.sh test — 전부 실행 · 원문 로그 · traceback 끝부분 · tests_summary.json"""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        tdir = tmp / 'tests'; tdir.mkdir()
        (tdir / 'test_ok.py').write_text("print('  OK a: got=1 want=1')\nprint('  OK b: got=2 want=2')\nprint('결과: 전부 통과')\n")
        (tdir / 'test_crash.py').write_text("print('  OK c: got=1 want=1')\nraise KeyError('score.json 에 visibility 키 없음')\n")
        env = dict(os.environ, FB_TEST_DIR=str(tdir), FB_RUNS=str(tmp / 'runs'), FB_LOGS=str(tmp / 'logs'))
        env.pop('MERIDIAN_WS', None) if 'MERIDIAN_WS' not in os.environ else None
        (tmp / 'runs').mkdir(); (tmp / 'logs').mkdir()
        r = subprocess.run(['bash', str(code_dir / 'eval.sh'), 'test'], capture_output=True, text=True, env=env, timeout=1800)
        out = r.stdout + r.stderr
        print('     ' + '\n     '.join(out.strip().splitlines()[-8:]))
        check('종료 코드 ≠ 0 (죽은 테스트가 있음)', r.returncode != 0, True)
        check('traceback 끝부분이 보임 (KeyError)', 'KeyError' in out, True)
        js = tmp / 'runs' / 'tests_summary.json'
        check('RUNS/tests_summary.json 생성', js.exists(), True)
        if js.exists():
            j = json.loads(js.read_text())
            f = j.get('files', {})
            check('test_ok.py 통과 · OK 2', (f.get('test_ok.py', {}).get('status'), f.get('test_ok.py', {}).get('n_ok')), ('pass', 2))
            check('test_crash.py 실패', f.get('test_crash.py', {}).get('status'), 'fail')
            check('합계 파일 2 · 통과 1', (j.get('totals', {}).get('files'), j.get('totals', {}).get('passed')), (2, 1))
            logp = f.get('test_crash.py', {}).get('log')
            check('원문 로그 파일에 traceback', bool(logp) and 'Traceback' in Path(logp).read_text(), True)


def test_c11_discovery(code_dir):
    """C11 run_tests.py --list — test_*.py 전부 · test_score_2d 는 자기 사례만"""
    r = subprocess.run([PY, str(code_dir / 'run_tests.py'), '--list'], capture_output=True, text=True, timeout=120,
                       env=dict(os.environ, FB_TEST_DIR=str(code_dir)))
    out = r.stdout
    print('     ' + '\n     '.join(out.strip().splitlines()[:20]))
    want = sorted(p.name for p in code_dir.glob('test_*.py'))
    listed = sorted(set(re.findall(r'(test_[A-Za-z0-9_]+\.py)', out)) & set(want))
    check('모든 test_*.py 가 목록에', listed, want)
    for n in ('test_trackeval_parity.py', 'test_gt_labels_2d.py', 'test_gt_difficulty_2d.py'):
        check(f'{n} 포함', n in out, True)
    line = next((l for l in out.splitlines() if 'test_score_2d.py' in l), '')
    check('[E4] test_score_2d.py 도 파일 그대로 실행 (main 이 자기 사례만 돈다 — own-only 우회 불필요)', 'own-only' in line, False)


def test_e4_score_2d_direct(code_dir):
    """E4 test_score_2d.py 를 직접 실행해도 라벨·난이도 테스트를 다시 돌리지 않는다"""
    r = subprocess.run([PY, str(code_dir / 'test_score_2d.py')], capture_output=True, text=True, timeout=600, cwd=str(code_dir))
    out = r.stdout + r.stderr
    check('종료 코드 0', r.returncode, 0)
    check('라벨(test_gt_labels_2d) 사례 없음', '== GT 라벨' in out, False)
    check('난이도(test_gt_difficulty_2d) 사례 없음', '== 난이도' in out, False)


def test_e4_real_paths(code_dir):
    """E4 실제 데이터 경로 · TrackEval 경로 기본값은 FB_RUNS 와 무관"""
    with tempfile.TemporaryDirectory() as d:
        env = dict(os.environ, FB_RUNS=str(Path(d) / 'empty_runs'))
        for k in ('FB_REAL_RUNS', 'TRACKEVAL_PATH', 'MERIDIAN_WS'):
            env.pop(k, None)
        code = ('import json, paths; print(json.dumps(dict(runs=str(paths.RUNS), real=str(paths.REAL_RUNS), '
                'te=str(paths.TRACKEVAL_PATH), ws=str(paths.WS))))')
        r = subprocess.run([PY, '-c', code], capture_output=True, text=True, timeout=120, cwd=str(code_dir), env=env)
        try:
            j = json.loads(r.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            print(r.stdout[-800:], r.stderr[-1500:]); j = {}
        check('RUNS = FB_RUNS (출력 폴더)', j.get('runs'), str(Path(d) / 'empty_runs'))
        check('REAL_RUNS = <WS>/runs/frontend_eval (FB_RUNS 무시)', j.get('real'), str(Path(j.get('ws', '?')) / 'runs/frontend_eval'))
        check('TRACKEVAL_PATH 가 FB_RUNS 아래가 아님', str(j.get('te', '')).startswith(str(Path(d))), False)


def test_e4_skip_reporting(code_dir):
    """E4 실제 데이터 없음 → 세 테스트 모두 SKIP(이유) · 종료 0 · run_tests 가 skip 으로 보고 · OK+SKIP 섞이면 pass 와 n_skip"""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        (tmp / 'runs').mkdir(); (tmp / 'logs').mkdir(); (tmp / 'no_real').mkdir()
        env = dict(os.environ, FB_RUNS=str(tmp / 'runs'), FB_LOGS=str(tmp / 'logs'), FB_REAL_RUNS=str(tmp / 'no_real'),
                   TRACKEVAL_PATH=str(tmp / 'no_trackeval'), FB_TEST_DIR=str(code_dir))
        names = ['test_report_cache.py', 'test_check_alignment.py', 'test_trackeval_parity.py']
        r = subprocess.run([PY, str(code_dir / 'run_tests.py'), '--only', *names], capture_output=True, text=True, timeout=900, env=env)
        out = r.stdout + r.stderr
        print('     ' + '\n     '.join(out.strip().splitlines()[-6:]))
        check('run_tests 종료 코드 0 (실패 없음)', r.returncode, 0)
        js = tmp / 'runs' / 'tests_summary.json'
        j = json.loads(js.read_text()) if js.exists() else {}
        for n in names:
            f = (j.get('files') or {}).get(n, {})
            log = Path(f['log']).read_text() if f.get('log') and Path(f['log']).exists() else ''
            check(f'{n}: status skip', f.get('status'), 'skip')
            check(f'{n}: SKIP 줄에 이유', bool(re.search(r'^\s*SKIP\b.+\S', log, re.M)), True)
            check(f'{n}: 예외 없음 (Traceback 없음)', 'Traceback' in log, False)
            check(f'{n}: skips 에 이유 기록', bool(f.get('skips')), True)
        check('합계 skipped 3', (j.get('totals') or {}).get('skipped'), 3)
        tdir = tmp / 'fake'; tdir.mkdir()
        (tdir / 'test_mixed.py').write_text("print('  OK a: got=1 want=1')\nprint('  SKIP 실제 데이터 없음')\nprint('결과: 전부 통과')\n")
        (tdir / 'test_skiponly.py').write_text("print('  SKIP 실제 데이터 없음 (예시)')\n")
        env2 = dict(env, FB_TEST_DIR=str(tdir))
        r = subprocess.run([PY, str(code_dir / 'run_tests.py')], capture_output=True, text=True, timeout=300, env=env2)
        j = json.loads(js.read_text()) if js.exists() else {}
        f = j.get('files') or {}
        check('OK+SKIP 섞인 파일: pass · n_skip 1', (f.get('test_mixed.py', {}).get('status'), f.get('test_mixed.py', {}).get('n_skip')), ('pass', 1))
        check('SKIP 만 있는 파일: skip', f.get('test_skiponly.py', {}).get('status'), 'skip')
        check('합계 n_skip 2', (j.get('totals') or {}).get('n_skip'), 2)


def main():
    code_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE
    for fn in (test_c7_stale_alignment, test_c7_gate_identity, test_c7_fresh_ok, test_c11_discovery, test_c11_eval_test,
               test_e4_score_2d_direct, test_e4_real_paths, test_e4_skip_reporting, test_f1_frontend_provenance):
        print(f'[{fn.__name__}] {fn.__doc__.strip()}')
        try:
            fn(code_dir)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc(limit=3)
            print(f'  FAIL {fn.__name__}: 예외 {type(e).__name__}: {e}')
            FAILS.append(fn.__name__)
    print('\n결과:', '전부 통과' if not FAILS else f'실패 {len(FAILS)}개 {FAILS}')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
