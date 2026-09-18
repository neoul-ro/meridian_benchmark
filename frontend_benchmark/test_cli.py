#!/usr/bin/env python3
"""eval.sh 명령줄 검증 (사용성 리뷰 C3 · M8 · M9 · M10 · m5). GPU·실제 결과를 건드리지 않는다.

가짜 코드 폴더(eval.sh · paths.py 사본 + 가짜 report.py · status_md.py · build_viewer.py …)에서 돌린다.
FB_RUNS · FB_LOGS · FB_DATA · FB_GT · FB_MODELS 는 임시 폴더, 터미널이 아니고(stdin=/dev/null) GPU=0 이다.

C3  모르는 명령·모호한 이름은 계획이나 GPU 단계 전에 종료 코드 2 로 멈추고 비슷한 명령을 알려 준다
    (오타 · 대문자 · --dry-run 을 시퀀스로 읽음 · 모호한 이름 · GPU=1 + 오타)
M8  채점 중 단계가 실시간으로 보인다 · Ctrl+C(종료 코드 130)는 traceback 없이 '중단됨 — 이어서' 안내
M9  정확한 이름 우선 · 후보만 나열 · 모든 하위 명령이 여러 시퀀스 · 도움말(명령 · 옵션 · 환경변수 · 종료 코드 · 예시) ·
    --dry-run/--gpu/--force 플래그 (환경변수도 그대로) · sh eval.sh 는 'bash 로 실행' 안내
M10 원인별 다음 행동 · 존댓말 · 빈 상태의 show/status 는 내부 스크립트 이름 대신 eval.sh 명령을 권한다
m5  local.env (FB_AUTHOR) 를 eval.sh 가 읽는다 · .gitignore 에 들어 있다
재리뷰 (logs2)
N1  status.md 사본(FB_STATUS_COPY)은 기본 결과 폴더의 전체 시퀀스 실행일 때만 · 빈 값이면 끄기 (환경변수가 local.env 를 이긴다)
N2  eval.sh test 가 status.md 의 '자체 검증 N항목' 도 같이 갱신한다
N4  결과가 없는 폴더에서 viewer 는 traceback 없이 한 줄로 멈춘다
N6  examples 는 같은 이유를 6번 반복하지 않는다
N8  화면에 보이는 시퀀스 이름은 짧은 쪽 하나로 (apartment_02h)
N9  읽기만 하는 명령(help · show · status · compare · history)은 폴더를 만들지 않는다
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parent
sys.path.insert(0, str(HERE))
FAILS = []


def check(name, got, want):
    ok = got == want
    print(f'  {"OK " if ok else "FAIL"} {name}: got={got!r} want={want!r}')
    if not ok:
        FAILS.append(name)


FAKE_REPORT = '''#!/usr/bin/env python3
import os, sys, time
mode = os.environ.get('FAKE_REPORT', 'ok')
if mode == 'slow':
    for i in range(3):
        print(f'[report] apartment_s1_00h step{i} 다시 계산 — 기록 없음', flush=True)
        time.sleep(0.4)
    print('[report] 다시 계산한 단계 3개', flush=True)
elif mode == 'interrupt':
    print('[report] apartment_s1_00h labels 다시 계산 — 기록 없음', flush=True)
    print('중단됨 (Ctrl+C)', file=sys.stderr, flush=True)
    sys.exit(130)
elif mode == 'stale':
    print('[report] 경고: apartment_s1_00h 의 frontend 출력이 지금 코드보다 오래됐습니다', flush=True)
    sys.exit(4)
elif mode == 'missing':
    print('[report] apartment_s1_00h 누락 — GT h5 없음 (/nowhere/gt.h5)', flush=True)
    sys.exit(2)
print('[report] 끝', flush=True)
open(os.environ['FB_RUNS'] + '/summary.md', 'w').write('## 1. 한눈에\\n\\n표\\n')
'''

FAKE_STATUS = '''#!/usr/bin/env python3
import os, sys
print('[status] ' + os.environ.get('FB_AUTHOR', '(작성자 없음)'))
print('[status] 인자 ' + ' '.join(sys.argv[1:]))
open(os.environ['FB_RUNS'] + '/status_called.txt', 'a').write(' '.join(sys.argv[1:]) + '\\n')
'''

FAKE_VIEWER = '''#!/usr/bin/env python3
import os, sys
print('[viewer] 인자 ' + ' '.join(sys.argv[1:]))
d = os.environ['FB_RUNS'] + '/viewer'
os.makedirs(d, exist_ok=True)
open(d + '/index.html', 'w').write('<html>목록</html>')
'''

FAKE_EXAMPLES = '''#!/usr/bin/env python3
import sys
print('[examples] 인자 ' + ' '.join(sys.argv[1:]))
'''

FAKE_TESTS = '''#!/usr/bin/env python3
import json, os, sys
print('[tests] 인자 ' + ' '.join(sys.argv[1:]))
json.dump(dict(totals=dict(files=1, passed=1, failed=0, skipped=0, n_ok=777, n_fail=0, n_skip=0)),
          open(os.environ['FB_RUNS'] + '/tests_summary.json', 'w'))
'''


def fake_code(tmp, with_local_env=True):
    """eval.sh 와 paths.py 사본 + 가짜 파이썬 스크립트가 있는 코드 폴더."""
    d = tmp / 'code'
    d.mkdir(parents=True, exist_ok=True)
    for n in ('eval.sh', 'paths.py', 'frontend_provenance.py', 'provenance.py'):
        shutil.copy2(HERE / n, d / n)
    (d / 'eval.sh').chmod(0o755)
    (d / 'report.py').write_text(FAKE_REPORT)
    (d / 'status_md.py').write_text(FAKE_STATUS)
    (d / 'build_viewer.py').write_text(FAKE_VIEWER)
    (d / 'render_examples.py').write_text(FAKE_EXAMPLES)
    (d / 'run_tests.py').write_text(FAKE_TESTS)
    (d / 'compare.py').write_text('#!/usr/bin/env python3\nimport sys\nprint("[compare] 인자 " + " ".join(sys.argv[1:]))\n')
    if with_local_env:
        (d / 'local.env').write_text('FB_AUTHOR=테스트작성자\n')
    return d


def seed_results(env, *seqs, score=True):
    """뷰어·판정 그림 사전 확인(N4 · N6)을 통과할 만큼의 가짜 결과를 만든다."""
    runs = Path(env['FB_RUNS'])
    for s in seqs:
        d = runs / f'uHumans2_{s}'
        d.mkdir(parents=True, exist_ok=True)
        (d / 'frontend_output.h5').write_bytes(b'fake')
        if score:
            (d / 'score.json').write_text('{}')
    return env


def sandbox_env(tmp, **extra):
    for n in ('runs', 'logs', 'models', 'data', 'gt'):
        (tmp / n).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, FB_RUNS=str(tmp / 'runs'), FB_LOGS=str(tmp / 'logs'), FB_MODELS=str(tmp / 'models'),
               FB_DATA=str(tmp / 'data'), FB_GT=str(tmp / 'gt'), GPU='0', CUDA_VISIBLE_DEVICES='')
    for k in ('DRY_RUN', 'FORCE', 'FB_AUTHOR', 'FB_ACCEPT_STALE'):
        env.pop(k, None)
    env.update({k: str(v) for k, v in extra.items()})
    return env


def run_sh(code, args, env, timeout=120, shell='bash'):
    r = subprocess.run([shell, str(code / 'eval.sh'), *args], capture_output=True, text=True, timeout=timeout,
                       stdin=subprocess.DEVNULL, env=env, cwd=str(WS))
    return r.returncode, r.stdout + r.stderr


# ------------------------------------------------------------------ C3 모르는 명령
def test_c3_unknown_command(tmp):
    """C3 오타 · 대문자 · 모호한 이름은 GPU 계획 전에 종료 코드 2 · '비슷한 명령' 안내"""
    code = fake_code(tmp / 'c3')
    env = sandbox_env(tmp / 'c3')
    for name, args, want_hint in (('오타 scroe', ['scroe'], 'score'), ('대문자 Score', ['Score'], 'score'),
                                  ('오타 vieweer', ['vieweer'], 'viewer')):
        rc, out = run_sh(code, args, env)
        check(f'{name}: 종료 코드 2', rc, 2)
        check(f'{name}: 비슷한 명령 {want_hint} 안내', want_hint in out, True)
        check(f'{name}: GPU 계획을 보이지 않음', '계획 — GPU' in out, False)
        check(f'{name}: 도움말 안내', 'help' in out, True)
    rc, out = run_sh(code, ['scroe'], sandbox_env(tmp / 'c3', GPU='1'))
    check('GPU=1 + 오타: 종료 코드 2', rc, 2)
    check('GPU=1 + 오타: GPU 단계 없음', ('계획 — GPU' in out) or ('frontend 다시' in out), False)
    rc, out = run_sh(code, ['score', '00h'], env)
    check('모호한 이름 00h: 종료 코드 2', rc, 2)
    check('모호한 이름 00h: 후보 2개만 (6개 전부 아님)', ('office_06h' in out, 'apartment_00h' in out, 'office_00h' in out),
          (False, True, True))
    rc, out = run_sh(code, ['score', '07h'], env)
    check('없는 이름 07h: 종료 코드 2', rc, 2)
    check('없는 이름 07h: 시퀀스 목록 안내', 'office_06h' in out, True)


def test_c3_dry_run_flag(tmp):
    """C3 · M9 --dry-run 은 시퀀스 이름이 아니라 플래그 (계획만, 종료 코드 0)"""
    code = fake_code(tmp / 'c3b')
    env = sandbox_env(tmp / 'c3b')
    rc, out = run_sh(code, ['--dry-run'], env)
    check('--dry-run: 시퀀스로 읽지 않음', '시퀀스 이름' in out, False)
    check('--dry-run: 종료 코드 0', rc, 0)
    check('--dry-run: 계획만', 'DRY_RUN' in out or '계획만' in out, True)
    rc, out = run_sh(code, ['--gpu', '--force', '--dry-run'], env)
    check('플래그 여러 개: 종료 코드 0', rc, 0)
    rc, out = run_sh(code, ['--nosuchflag'], env)
    check('모르는 옵션: 종료 코드 2', rc, 2)
    check('모르는 옵션: 안내', '--nosuchflag' in out, True)


# ------------------------------------------------------------------ M9 도움말 · 이름 · 여러 시퀀스
def test_m9_help(tmp):
    """M9 도움말에 명령 · 옵션 · 환경변수 · 종료 코드 · 예시가 있고 코드가 섞여 나오지 않는다"""
    code = fake_code(tmp / 'm9')
    env = sandbox_env(tmp / 'm9')
    rc, out = run_sh(code, ['help'], env)
    check('help: 종료 코드 0', rc, 0)
    for want in ('명령', '옵션', '환경변수', '종료 코드', '예시', 'score', 'compare', 'viewer', 'test', '--dry-run', '--force',
                 'FB_RUNS', 'GPU=1', '130'):
        check(f'help 에 {want}', want in out, True)
    check('help 에 셸 코드 조각이 섞이지 않음', ('BASH_SOURCE' in out) or ('set -euo' in out), False)
    rc2, out2 = run_sh(code, ['-h'], env)
    check('-h 도 같은 도움말', (rc2, out2 == out), (0, True))
    rc3, out3 = run_sh(code, ['--help'], env)
    check('--help 도 같은 도움말', (rc3, out3 == out), (0, True))


def test_m9_resolve_and_multi(tmp):
    """M9 정확한 이름 우선 · 모든 하위 명령이 여러 시퀀스를 받는다"""
    code = fake_code(tmp / 'm9b')
    env = seed_results(sandbox_env(tmp / 'm9b'), 'apartment_s1_00h', 'apartment_s1_01h', 'apartment_s1_02h',
                       'office_s1_00h', 'office_s1_06h', 'office_s1_12h')
    rc, out = run_sh(code, ['viewer', 'apartment_00h', 'office_06h'], env)
    check('viewer 두 시퀀스: 종료 코드 0', rc, 0)
    check('viewer 두 시퀀스 모두 전달', ('apartment_s1_00h' in out, 'office_s1_06h' in out), (True, True))
    check('viewer: 목록 페이지 경로 안내', 'viewer/index.html' in out, True)
    rc, out = run_sh(code, ['examples', 'apartment_00h', 'office_06h'], env)
    check('examples 두 시퀀스: 종료 코드 0', rc, 0)
    check('examples 두 시퀀스 모두 실행', (out.count('[examples]') >= 2), True)
    rc, out = run_sh(code, ['examples'], env)
    check('examples 인자 없음: 6개 전부 (종료 코드 0)', rc, 0)
    rc, out = run_sh(code, ['score', 'apartment_s1_00h', 'office_s1_06h'], env)
    check('score 두 시퀀스: 종료 코드 0', rc, 0)


def test_m9_sh_invocation(tmp):
    """M9 sh eval.sh → 'bash 로 실행' 안내 (Illegal option 아님)"""
    code = fake_code(tmp / 'm9c')
    env = sandbox_env(tmp / 'm9c')
    rc, out = run_sh(code, ['score'], env, shell='sh')
    check('sh 실행: bash 안내', 'bash' in out, True)
    check('sh 실행: Illegal option 메시지 없음', 'Illegal option' in out, False)
    check('sh 실행: 종료 코드 2', rc, 2)


# ------------------------------------------------------------------ M8 진행 표시 · 중단
def test_m8_progress_and_interrupt(tmp):
    """M8 채점 단계가 실시간으로 보이고, 중단(130)은 traceback 없이 이어서 하라고 안내한다"""
    code = fake_code(tmp / 'm8')
    env = sandbox_env(tmp / 'm8', FAKE_REPORT='slow')
    rc, out = run_sh(code, ['score'], env)
    check('진행 줄이 화면에', out.count('step') >= 3, True)
    check('채점 완료', rc, 0)
    env2 = sandbox_env(tmp / 'm8', FAKE_REPORT='interrupt')
    rc, out = run_sh(code, ['score'], env2)
    check('중단: 종료 코드 130', rc, 130)
    check('중단: 이어서 한다는 안내', '이어서' in out, True)
    check('중단: "실패" 로 표시하지 않음', '채점 실패' in out, False)


# ------------------------------------------------------------------ M10 안내 · 존댓말
def test_m10_advice(tmp):
    """M10 빈 상태의 show · status 는 eval.sh 명령을 권하고, 말투는 존댓말"""
    code = fake_code(tmp / 'm10')
    env = sandbox_env(tmp / 'm10')
    rc, out = run_sh(code, ['show'], env)
    check('show 빈 상태: 종료 코드 2', rc, 2)
    check('show 빈 상태: eval.sh score 안내', 'eval.sh score' in out, True)
    check('show 빈 상태: 반말 아님', bool(re.search(r'없다|할 것|한다\b', out)), False)
    rc, out = run_sh(code, ['score'], sandbox_env(tmp / 'm10', FAKE_REPORT='missing'))
    check('누락: 종료 코드 2', rc, 2)
    check('누락: 다음 행동 안내', ('FB_GT' in out) or ('확인' in out), True)
    env_stale = sandbox_env(tmp / 'm10', FAKE_REPORT='stale')
    rc, out = run_sh(code, ['score'], env_stale)
    check('오래된 frontend 출력: 종료 코드 4', rc, 4)
    check('오래된 frontend 출력: 받아들이는 방법 안내', '--accept-stale' in out, True)
    check('[N3] 종료 4 에서도 status.md 를 다시 만든다', (Path(env_stale['FB_RUNS']) / 'status_called.txt').exists(), True)


def test_m10_polite(tmp):
    """M10 CLI 문구가 존댓말로 통일 (eval.sh 안의 사용자 메시지)"""
    sh = (HERE / 'eval.sh').read_text()
    msgs = re.findall(r"(?:echo|printf)\s+(?:-e\s+)?['\"]([^'\"]*[가-힣][^'\"]*)['\"]", sh)
    bad = [m for m in msgs if re.search(r'(없다|한다|먼저$|볼 것|할 것|된다|같다)\s*$', m.strip())]
    check('eval.sh 사용자 메시지에 반말 없음', bad, [])


# ------------------------------------------------------------------ m5 local.env
def test_m5_local_env(tmp):
    """m5 local.env 의 FB_AUTHOR 를 eval.sh 가 읽어 status_md 에 넘긴다 · .gitignore 에 들어 있다"""
    code = fake_code(tmp / 'm5')
    env = sandbox_env(tmp / 'm5')
    rc, out = run_sh(code, ['status'], env)
    check('local.env 의 FB_AUTHOR 사용', '테스트작성자' in out, True)
    env2 = sandbox_env(tmp / 'm5', FB_AUTHOR='환경변수작성자')
    rc, out = run_sh(code, ['status'], env2)
    check('환경변수가 local.env 보다 우선', '환경변수작성자' in out, True)
    gi = HERE / '.gitignore'
    check('.gitignore 있음', gi.exists(), True)
    check('.gitignore 에 local.env', 'local.env' in (gi.read_text() if gi.exists() else ''), True)
    le = HERE / 'local.env'
    check('이 워크스페이스 local.env 에 FB_AUTHOR=김주영', 'FB_AUTHOR=김주영' in (le.read_text() if le.exists() else ''), True)


# ------------------------------------------------------------------ 재리뷰 N1 · N2 · N4 · N6 · N8 · N9
def test_n1_status_copy_scope(tmp):
    """N1 사본은 기본 결과 폴더의 전체 시퀀스 실행일 때만 · 빈 환경변수가 local.env 를 이긴다"""
    import subprocess as sp
    code = fake_code(tmp / 'n1')
    (code / 'local.env').write_text('FB_AUTHOR=테스트작성자\nFB_STATUS_COPY=/tmp/절대_없는_경로/status.md\n')
    env = sandbox_env(tmp / 'n1')
    rc, out = run_sh(code, ['status'], env)
    check('샌드박스 FB_RUNS: 사본 경로를 status_md 에 넘기지 않음', '/tmp/절대_없는_경로' in out, False)
    r = sp.run(['bash', '-c', f'set -a; source "{code}/local.env"; set +a; FB_STATUS_COPY= bash "{code}/eval.sh" status'],
               capture_output=True, text=True, env=env, stdin=sp.DEVNULL, cwd=str(WS), timeout=120)
    check('빈 FB_STATUS_COPY 가 local.env 를 이긴다', '/tmp/절대_없는_경로' in (r.stdout + r.stderr), False)
    sh = (HERE / 'eval.sh').read_text()
    check('eval.sh 가 "설정됨" 을 보고 덮어쓰지 않는다 (-v)', '-v $k' in sh or '-v "$k"' in sh, True)
    import status_md as SMD
    import paths as P
    full = {s: {} for s in P.SEQUENCES}
    part = {list(full)[0]: {}}
    check('전체 시퀀스 + 기본 폴더 → 사본 만든다', bool(SMD.copy_targets(full, runs=SMD.default_runs(), value='/x/a.md')[0]), True)
    check('부분 실행 → 사본 안 만든다', SMD.copy_targets(part, runs=SMD.default_runs(), value='/x/a.md')[0], [])
    check('다른 결과 폴더(샌드박스) → 사본 안 만든다', SMD.copy_targets(full, runs=Path('/tmp/sandbox_runs'), value='/x/a.md')[0], [])
    check('건너뛴 이유를 알려 준다', bool(SMD.copy_targets(part, runs=SMD.default_runs(), value='/x/a.md')[1]), True)
    check('--copy-to 는 명시 요청이라 그대로 따른다', SMD.copy_targets(part, runs=Path('/tmp/x'), value='/x/a.md', explicit=True)[0], [Path('/x/a.md')])
    rd = (HERE / 'README.md').read_text()
    check('README 에 FB_STATUS_COPY', 'FB_STATUS_COPY' in rd, True)
    rc, out = run_sh(fake_code(tmp / 'n1h'), ['help'], sandbox_env(tmp / 'n1h'))
    check('help 에 FB_STATUS_COPY', 'FB_STATUS_COPY' in out, True)


def test_n2_test_refreshes_status(tmp):
    """N2 eval.sh test 가 status.md 의 자체 검증 항목 수도 갱신한다"""
    code = fake_code(tmp / 'n2')
    env = sandbox_env(tmp / 'n2')
    (Path(env['FB_RUNS']) / 'summary.json').write_text('{}')
    rc, out = run_sh(code, ['test'], env)
    called = Path(env['FB_RUNS']) / 'status_called.txt'
    check('test 뒤 status_md 를 부른다', called.exists(), True)
    check('test 종료 코드 0', rc, 0)


def test_n4_viewer_empty(tmp):
    """N4 결과가 없는 폴더의 viewer 는 traceback 없이 한 줄로 멈춘다"""
    code = fake_code(tmp / 'n4')
    env = sandbox_env(tmp / 'n4')
    rc, out = run_sh(code, ['viewer'], env)
    check('종료 코드 2', rc, 2)
    check('traceback 없음', 'Traceback' in out, False)
    check('무엇이 없는지 · 다음 행동', ('frontend 출력' in out or '채점 결과' in out) and 'eval.sh score' in out, True)


def test_n6_examples_repeat(tmp):
    """N6 examples 는 같은 이유를 시퀀스마다 반복하지 않는다"""
    code = fake_code(tmp / 'n6')
    env = sandbox_env(tmp / 'n6')
    rc, out = run_sh(code, ['examples'], env)
    check('종료 코드 2', rc, 2)
    check('같은 안내가 3번 이상 반복되지 않음', out.count('먼저 채점해 주세요') <= 2, True)
    check('몇 개 시퀀스가 걸렸는지 한 줄로', ('6개' in out) or ('6 개' in out), True)


def test_n8_short_names(tmp):
    """N8 화면에 보이는 시퀀스 이름은 짧은 쪽 하나 (apartment_02h)"""
    code = fake_code(tmp / 'n8')
    env = sandbox_env(tmp / 'n8')
    rc, out = run_sh(code, ['score', 'apartment_02h'], env)
    check('채점 머리말에 짧은 이름', 'apartment_02h' in out, True)
    check('채점 머리말에 긴 이름 없음', 'apartment_s1_02h' in out.split('== 채점')[1].split(chr(10))[0] if '== 채점' in out else False, False)
    rc, out = run_sh(code, ['score', '07h'], env)
    check('없는 이름 안내도 짧은 이름 목록', ('apartment_00h' in out) and ('apartment_s1_00h' not in out), True)
    rc, out = run_sh(code, ['score', '00h'], env)
    check('모호 안내도 짧은 이름', ('apartment_00h' in out) and ('apartment_s1_00h' not in out), True)


def test_n9_readonly_no_mkdir(tmp):
    """N9 읽기만 하는 명령은 폴더를 만들지 않는다"""
    code = fake_code(tmp / 'n9')
    for cmd in (['help'], ['show'], ['status'], ['history'], ['compare']):
        d = tmp / f'n9_{cmd[0]}'
        env = sandbox_env(d)
        runs, logs = Path(env['FB_RUNS']), Path(env['FB_LOGS'])
        for p in (runs, logs):
            shutil.rmtree(p, ignore_errors=True)
        run_sh(code, cmd, env)
        made = [str(x.relative_to(d)) for x in (runs, logs, runs / 'gt_vis') if x.exists()]
        check(f'{cmd[0]}: 새로 만든 폴더 없음', made, [])


def main():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for fn in (test_c3_unknown_command, test_c3_dry_run_flag, test_m9_help, test_m9_resolve_and_multi,
                   test_m9_sh_invocation, test_m8_progress_and_interrupt, test_m10_advice, test_m10_polite, test_m5_local_env,
                   test_n1_status_copy_scope, test_n2_test_refreshes_status, test_n4_viewer_empty, test_n6_examples_repeat,
                   test_n8_short_names, test_n9_readonly_no_mkdir):
            print(f'[{fn.__name__}] {fn.__doc__.strip().splitlines()[0]}')
            try:
                fn(tmp)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc(limit=3)
                print(f'  FAIL {fn.__name__}: 예외 {type(e).__name__}: {e}')
                FAILS.append(fn.__name__)
    print('\n결과:', '전부 통과' if not FAILS else f'실패 {len(FAILS)}개 {FAILS}')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
