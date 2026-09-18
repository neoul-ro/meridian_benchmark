"""경로는 여기 한 곳에서만 정한다. 다른 곳(meridian_benchmark 저장소 등)으로 옮기면 환경변수로 맞춘다.

  MERIDIAN_WS   워크스페이스 루트 (기본: 이 폴더의 부모)
  FB_DATA       uHumans2 언팩본       (기본: $WS/datasets/unpacked)
  FB_GT         GT tracklet h5 폴더   (기본: $WS/src/meridian/meridian_benchmark/tracklets)
  FB_BENCH_PKG  meridian_benchmark 패키지 경로 — uhumans2 리더를 쓴다
  FB_RUNS       실행·채점 결과         (기본: $WS/runs/frontend_eval)
  FB_LOGS       로그                   (기본: $WS/logs/frontend_eval)
  FB_MODELS     ONNX·엔진              (기본: $WS/models)
  FB_REAL_RUNS  테스트가 읽는 실제 결과 (기본: $WS/runs/frontend_eval) — FB_RUNS(출력·샌드박스)를 바꿔도 따라가지 않는다 (수정 ⑤ E4).
                없으면 실제 데이터 테스트는 'SKIP <이유>' 로 건너뛴다
  FB_FRONTEND_SRC · FB_FRONTEND_MSGS  frontend 소스 트리 (기본: $WS/src/meridian_frontend · $WS/src/meridian_frontend_msgs) —
                frontend 를 다시 돌릴지 정하는 출처 기록(frontend_provenance.py)에 내용 해시로 들어간다
  TRACKEVAL_PATH TrackEval 저장소 (기본: $WS/third_party/TrackEval, commit 12c8791) — 찾는 순서는 trackeval_path.py
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = Path(os.environ.get('MERIDIAN_WS', HERE.parent))
DATA = Path(os.environ.get('FB_DATA', WS / 'datasets/unpacked'))
GT_DIR = Path(os.environ.get('FB_GT', WS / 'src/meridian/meridian_benchmark/tracklets'))
BENCH_PKG = Path(os.environ.get('FB_BENCH_PKG', WS / 'src/meridian/meridian_benchmark'))
RUNS = Path(os.environ.get('FB_RUNS', WS / 'runs/frontend_eval'))
LOGS = Path(os.environ.get('FB_LOGS', WS / 'logs/frontend_eval'))
MODELS = Path(os.environ.get('FB_MODELS', WS / 'models'))
REAL_RUNS = Path(os.environ.get('FB_REAL_RUNS', WS / 'runs/frontend_eval'))
FRONTEND_SRC = Path(os.environ.get('FB_FRONTEND_SRC', WS / 'src/meridian_frontend'))
FRONTEND_MSGS = Path(os.environ.get('FB_FRONTEND_MSGS', WS / 'src/meridian_frontend_msgs'))
TRACKEVAL_PATH = Path(os.environ.get('TRACKEVAL_PATH', WS / 'third_party/TrackEval'))

SEQUENCES = ['apartment_s1_00h', 'apartment_s1_01h', 'apartment_s1_02h',
             'office_s1_00h', 'office_s1_06h', 'office_s1_12h']

if str(BENCH_PKG) not in sys.path:
    sys.path.insert(0, str(BENCH_PKG))


def seq_dir(s):
    return DATA / f'uHumans2_{s}'


def gt_h5(s):
    return GT_DIR / f'uHumans2_{s}.h5'


def run_dir(s):
    return RUNS / f'uHumans2_{s}'


def vis_npz(s):
    return RUNS / 'gt_vis' / f'uHumans2_{s}.npz'


def shell_exports():
    """eval.sh 가 경로를 여기 한 곳에서 받도록 셸 변수 대입문으로 (FB_* 환경변수 반영). 26/09/17 추가 (감사 C7·C9 — 샌드박스 실행)."""
    import shlex
    return '\n'.join(f'{k}={shlex.quote(str(v))}' for k, v in (
        ('WS', WS), ('DATA', DATA), ('GT_DIR', GT_DIR), ('BENCH_PKG', BENCH_PKG), ('RUNS', RUNS), ('LOGS', LOGS), ('MODELS', MODELS),
        ('FRONTEND_SRC', FRONTEND_SRC), ('FRONTEND_MSGS', FRONTEND_MSGS)))


if __name__ == '__main__':
    if '--sh' in sys.argv:
        print(shell_exports())
    else:
        print(__doc__)
