#!/usr/bin/env python3
"""채점기 자체 검증을 전부 돌린다 (eval.sh test, 감사 C11).

  test_*.py 를 전부 찾아(폴더 = frontend_benchmark/tests, FB_TEST_DIR 또는 --dir 로 바꿈) 파일마다 따로 실행한다.
  원문 로그   <LOGS>/tests/<파일 이름>.log  — 필터 없이 전부 (예전 eval.sh 는 grep 으로 Traceback 을 삼켰다)
  요약        <RUNS>/tests_summary.json
              files.<파일>: status(pass|fail|skip) · exit_code · n_ok('  OK ' 줄 수) · n_fail('FAIL' 줄 수) · n_skip('SKIP' 줄 수) ·
                            skips(SKIP 줄 = 건너뛴 이유) · seconds · log · mode · result
              totals: files · passed · failed · skipped · n_ok · n_fail · n_skip
  판정        pass = 종료 코드 0 이고 FAIL 줄 0 (SKIP 줄이 섞여 있으면 n_skip 으로 따로 센다)
              skip = 종료 코드 0, OK 0, 'SKIP' 줄 있음 (실제 데이터가 없어 파일 전체를 건너뜀 — 통과로 세지 않는다)
              나머지 fail
  실패하면    로그 끝 --tail 줄(traceback 포함)을 화면에 보인다. 하나라도 fail 이면 종료 코드 1.
  실제 데이터  테스트는 paths.REAL_RUNS(FB_REAL_RUNS) · paths.TRACKEVAL_PATH 에서 읽는다 — FB_RUNS 는 이 러너의 출력 위치일 뿐 (수정 ⑤ E4)
실행: python run_tests.py [--list] [--dir D] [--only test_a.py ...] [--tail 40]   (기본 폴더 = frontend_benchmark/tests)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths as P  # noqa: E402

def plan(d, only=None):
    """test_*.py 를 파일마다 그대로 실행 (수정 ⑤ E4: test_score_2d.py 의 main 이 자기 사례만 돌아서 own-only 우회가 필요 없다)."""
    files = sorted(p.name for p in Path(d).glob('test_*.py'))
    if only:
        files = [f for f in files if f in only]
    return [dict(file=f, mode='file', note='', cmd=[sys.executable, '-u', str(Path(d) / f)]) for f in files]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.environ.get('FB_TEST_DIR', str(HERE / 'tests')),
                    help='test_*.py 가 있는 폴더 (기본 frontend_benchmark/tests)')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--tail', type=int, default=40)
    ap.add_argument('--timeout', type=int, default=3600)
    a = ap.parse_args()
    d = Path(a.dir).resolve()
    items = plan(d, a.only)
    if a.list:
        for it in items:
            print(f'{it["file"]:32s} {it["mode"]}{"  (" + it["note"] + ")" if it["note"] else ""}')
        return 0
    logs = P.LOGS / 'tests'; logs.mkdir(parents=True, exist_ok=True)
    files = {}
    t_all = time.time()
    for it in items:
        name = it['file']; log = logs / f'{name[:-3]}.log'
        print(f'-- {name} ({it["mode"]})', flush=True)
        t0 = time.time()
        try:
            r = subprocess.run(it['cmd'], cwd=str(d), capture_output=True, text=True, timeout=a.timeout)
            out, code = r.stdout + r.stderr, r.returncode
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or '') + (e.stderr or '') if isinstance(e.stdout, str) else ''
            out += f'\nTIMEOUT {a.timeout}s'; code = 124
        log.write_text(out)
        n_ok = len(re.findall(r'^\s*OK\b', out, re.M))
        n_fail = len(re.findall(r'^\s*FAIL\b', out, re.M))
        skips = [l.strip() for l in re.findall(r'^\s*SKIP\b.*$', out, re.M)]
        skipped = code == 0 and n_ok == 0 and n_fail == 0 and bool(skips)
        status = 'skip' if skipped else ('pass' if code == 0 and n_fail == 0 else 'fail')
        result = next((l.strip() for l in reversed(out.splitlines()) if l.strip().startswith('결과')), '')
        files[name] = dict(status=status, exit_code=code, n_ok=n_ok, n_fail=n_fail, n_skip=len(skips), skips=skips,
                           seconds=round(time.time() - t0, 1), log=str(log), mode=it['mode'], result=result)
        mark = {'pass': '통과', 'fail': '실패', 'skip': '건너뜀'}[status]
        print(f'   {mark} · OK {n_ok} · FAIL {n_fail} · SKIP {len(skips)} · 종료 {code} · {files[name]["seconds"]}s', flush=True)
        for sk in skips:
            print(f'   · {sk}', flush=True)
        if status != 'pass':
            tail = out.rstrip().splitlines()[-a.tail:]
            print('   ── 로그 끝 (' + str(log) + ')\n' + '\n'.join('   | ' + l for l in tail), flush=True)
    totals = dict(files=len(files), passed=sum(v['status'] == 'pass' for v in files.values()),
                  failed=sum(v['status'] == 'fail' for v in files.values()), skipped=sum(v['status'] == 'skip' for v in files.values()),
                  n_ok=sum(v['n_ok'] for v in files.values()), n_fail=sum(v['n_fail'] for v in files.values()),
                  n_skip=sum(v['n_skip'] for v in files.values()), seconds=round(time.time() - t_all, 1))
    summ = dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'), python=sys.executable, dir=str(d), files=files, totals=totals)
    out = P.RUNS / 'tests_summary.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summ, indent=2, ensure_ascii=False))
    print(f'\n결과: 파일 {totals["files"]}개 중 통과 {totals["passed"]} · 실패 {totals["failed"]} · 건너뜀 {totals["skipped"]} · '
          f'OK {totals["n_ok"]} · FAIL {totals["n_fail"]} · SKIP 줄 {totals["n_skip"]} · {totals["seconds"]}s → {out}')
    sk = {n: v['skips'] for n, v in files.items() if v['status'] == 'skip'}
    if sk:                                                     # M10: 건너뛴 이유와 다음 행동을 눈에 보이게
        print(f'건너뛴 파일 {len(sk)}개 — 실제 결과나 TrackEval 사본이 없으면 건너뜁니다:')
        for n, why in sk.items():
            print(f'  · {n}: {why[0] if why else "이유 없음"}')
        print('  채점 결과가 필요하면 `bash frontend_benchmark/eval.sh score` 를, TrackEval 이 필요하면 '
              '`python frontend_benchmark/trackeval_path.py` 의 안내를 따라 주세요.')
    if totals['failed']:
        print(f'실패한 파일 {totals["failed"]}개의 원문 로그는 {logs} 에 있습니다.')
    return 1 if totals['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
