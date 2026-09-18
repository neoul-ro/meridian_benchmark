#!/usr/bin/env python3
"""TrackEval(추적 지표 표준 구현) 을 어디서 찾을지 정한다 — 대조 테스트(test_trackeval_parity.py)가 쓴다. (사용성 리뷰 M7)

왜 따로 두나: 예전에는 `paths.TRACKEVAL_PATH` 기본값이 이 워크스페이스의 감사 폴더를 가리켰다. 팀 저장소를 받은 사람에게는
  그 경로가 없어서 대조 테스트가 조용히 SKIP 됐다. 기본값에서 내부 폴더를 빼고, 받는 법을 코드와 문서에 같이 적는다.
찾는 순서
  1. 환경변수 TRACKEVAL_PATH (frontend_benchmark/local.env 에 적어 둬도 된다 — 사람마다 다른 설정)
  2. pip 로 설치된 `trackeval` 패키지 (import 되면 그 폴더)
  3. <WS>/third_party/TrackEval  (권장 위치)
  없으면 None 과 함께 받는 법(HOWTO)을 알려 준다 — 테스트는 이유를 적고 SKIP 한다.
받는 법 (고정 commit 12c8791 — 우리 결과를 만든 판)
  pip install 'git+https://github.com/JonathonLuiten/TrackEval@12c8791'
  또는  git clone https://github.com/JonathonLuiten/TrackEval <WS>/third_party/TrackEval && git -C <WS>/third_party/TrackEval checkout 12c8791
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402

COMMIT = '12c8791'
HOWTO = (f"TrackEval(commit {COMMIT}) 을 받아 주세요: "
         f"pip install 'git+https://github.com/JonathonLuiten/TrackEval@{COMMIT}' "
         f"또는 git clone 후 {P.WS}/third_party/TrackEval 에 두고 {COMMIT} 로 checkout, "
         f"또는 이미 있는 사본 경로를 TRACKEVAL_PATH 에 넣어 주세요 (frontend_benchmark/local.env 에 적어도 됩니다).")


def local_env(name):
    """frontend_benchmark/local.env 의 값 (사람마다 다른 설정, git 에 올리지 않는다)."""
    p = Path(__file__).resolve().parent / 'local.env'
    try:
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line and line.split('=', 1)[0].strip() == name:
                return line.split('=', 1)[1].strip()
    except OSError:
        pass
    return None


def pip_package():
    """pip 로 설치된 trackeval 패키지 폴더 (없으면 None)."""
    try:
        import trackeval  # noqa: F401
    except Exception:  # noqa: BLE001
        return None
    f = getattr(sys.modules['trackeval'], '__file__', None)
    return Path(f).resolve().parent.parent if f else None


def defaults():
    """코드가 기본으로 보는 곳 (환경변수 · local.env 없이). 내부 감사 폴더는 넣지 않는다."""
    return [Path(P.WS) / 'third_party' / 'TrackEval']


def resolve(use_env=True):
    """→ (경로 또는 None, 어디서 왔는지 한 줄). import 는 test_trackeval_parity.py 가 한다."""
    if use_env:
        v = os.environ.get('TRACKEVAL_PATH') or local_env('TRACKEVAL_PATH')
        if v:
            return Path(v), 'TRACKEVAL_PATH 설정'
    pkg = pip_package()
    if pkg is not None:
        return pkg, 'pip 로 설치된 trackeval 패키지'
    for d in defaults():
        if (d / 'trackeval').is_dir() or (d / 'setup.py').exists():
            return d, '기본 위치 <WS>/third_party/TrackEval'
    return None, '없음'


if __name__ == '__main__':
    p, how = resolve()
    print(f'TrackEval: {p or "없음"} ({how})')
    if p is None:
        print(HOWTO)
