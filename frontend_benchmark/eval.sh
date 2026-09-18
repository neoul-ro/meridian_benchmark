#!/usr/bin/env bash
# frontend 평가 진입점. 도움말은 `bash frontend_benchmark/eval.sh help` (아래 usage 함수).
# 환경(ROS+venv)은 이 스크립트가 잡고, 경로는 paths.py 한 곳에서 온다 (FB_RUNS·FB_LOGS·FB_GT·FB_DATA·FB_MODELS 로 바꿈).
# 이 워크스페이스 전용 설정은 frontend_benchmark/local.env (git 에 올리지 않음, 환경변수가 우선).
#
# GPU 단계 (수정 ⑤ F1): frontend 는 출처 기록(run_meta.json provenance — run_frontend.py 의 docstring 뺀 AST · frontend 소스 내용 ·
#   엔진 sha1 · 데이터셋 경로, frontend_provenance.py)이 지금과 다를 때만 다시 돈다. docstring·주석만 고친 것은 다시 돌 이유가 아니다.
#   다시 돌려야 하면 GPU=1(또는 --gpu)이거나 터미널에서 y 라고 답해야 진행하고, 아니면 아무 단계도 돌기 전에 안내를 보이고 종료 코드 3.
# CPU 단계마다 산출물이 없거나, 입력(코드·데이터·앞 단계 산출물) 중 하나라도 산출물보다 새로우면(-nt) 다시 만든다 (감사 C7).
# 정렬 게이트는 alignment_check.json 에 기록된 frontend_output.h5(크기·mtime, 다르면 sha1)가 지금 파일과 같고 gate.pass 일 때만 통과.
# 채점(report.py)의 캐시는 내용(sha1) 기반 — report.py 상단 참고. 채점할 때마다 이전 결과를 <RUNS>/history/ 에 보관한다 (compare).

if [ -z "${BASH_VERSION:-}" ]; then                # sh eval.sh 로 부른 경우 (M9)
  echo "이 스크립트는 bash 로 실행해 주세요:" >&2
  echo "  bash frontend_benchmark/eval.sh score" >&2
  exit 2
fi
set -euo pipefail

HERE=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd -P)   # 심볼릭 링크로 불려도 실제 위치
WS=${MERIDIAN_WS:-$(cd "$HERE/.." && pwd -P)}
cd "$WS"

load_local_env() {               # frontend_benchmark/local.env (KEY=VALUE) — 이미 있는 환경변수가 우선 (m5)
  local line k v
  [[ -f $HERE/local.env ]] || return 0
  while IFS= read -r line || [[ -n $line ]]; do
    line=${line%%$'\r'}
    [[ -z ${line// } || ${line:0:1} == '#' || $line != *=* ]] && continue
    k=${line%%=*}; v=${line#*=}
    k=${k// /}
    [[ -v $k ]] || export "$k=$v"     # 이미 설정돼 있으면 (빈 값이어도) 환경변수가 이깁니다 (재리뷰 N1)
  done < "$HERE/local.env"
}
load_local_env

set +u                            # ROS/colcon setup 스크립트가 미정의 변수를 참조한다
if [[ -f $WS/tools/env.sh ]]; then
  source "$WS/tools/env.sh" >/dev/null 2>&1
else
  echo "경고: $WS/tools/env.sh 가 없습니다 — ROS·venv 환경을 잡지 못했습니다 (MERIDIAN_WS 를 확인해 주세요)" >&2
fi
set -u
PY=python3
export PYTHONWARNINGS=${PYTHONWARNINGS:-ignore::UserWarning}   # scipy 의 numpy 버전 경고 줄 (필터 대신)
eval "$($PY "$HERE/paths.py" --sh)"                             # WS DATA GT_DIR BENCH_PKG RUNS LOGS MODELS
OUT=$RUNS; LOG=$LOGS; GTDIR=$GT_DIR
need_dirs() {                    # 무언가를 쓰는 명령만 폴더를 만듭니다 (재리뷰 N9 — help·show·status·compare·history 는 아무것도 만들지 않습니다)
  mkdir -p "$LOG" "$OUT/gt_vis"
}
FORCE=${FORCE:-0}
GPU=${GPU:-0}; DRY_RUN=${DRY_RUN:-0}; GPU_APPROVED=0
ACCEPT_STALE=${FB_ACCEPT_STALE:-0}

ALL_SEQS=(apartment_s1_00h apartment_s1_01h apartment_s1_02h office_s1_00h office_s1_06h office_s1_12h)
COMMANDS=(score viewer compare history status show examples test engines run all help)

short_seq() {                    # 화면에 보이는 이름은 짧은 쪽 하나 (apartment_02h). 폴더 이름은 uHumans2_<시퀀스> (재리뷰 N8)
  local s out=()
  for s in "$@"; do out+=("${s//_s1_/_}"); done
  echo "${out[*]}"
}

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()  { printf '   \033[32m✓\033[0m %s\n' "$*"; }
skip() { printf '   · %s (최신 — 다시 하려면 --force)\n' "$*"; }

usage() {                        # M9: 명령 · 옵션 · 환경변수 · 종료 코드 · 예시
  cat <<'EOF'
frontend 평가 (uHumans2) — 채점 · 뷰어 · 이전 결과와 비교

사용법
  bash frontend_benchmark/eval.sh <명령> [시퀀스 ...] [옵션]

명령
  score [시퀀스...]     채점만 합니다 (GPU 를 쓰지 않고 이미 있는 frontend 출력으로). 인자가 없으면 6개 전부
                        → runs/frontend_eval/summary.md (전체 표) · status.md (팀 공유용 요약)
  viewer [시퀀스...]    뷰어를 만듭니다 → runs/frontend_eval/viewer/index.html (목록 페이지)
  compare [id|latest]   이전 결과와 지금 결과를 견줍니다 (시퀀스 × 핵심 지표 변화표)
  history               보관된 이전 결과 목록을 보여 줍니다
  show                  마지막 채점표의 앞부분을 보여 줍니다
  status                status.md 만 다시 만듭니다 (summary.json · tests_summary.json 에서). --copy-to 로 사본 경로를 직접 줄 수 있습니다
  examples [시퀀스...]  판정 그림을 만듭니다 (입력이 바뀌었을 때만 다시 그립니다)
  test [파일...]        채점기 자체 검증을 돌립니다 (frontend_benchmark/tests/test_*.py). 파일 이름을 주면 그것만
                        (예: eval.sh test test_cli.py). 끝나면 status.md 의 검증 항목 수도 다시 씁니다
  engines               TensorRT 엔진을 다시 빌드합니다 (GPU, 약 3.5분)
  run <시퀀스...>       그 시퀀스의 frontend 를 돌리고(GPU) 정렬 검증 · 가시성 후 전체 채점 · 뷰어까지
  all                   전부 (기본값). 먼저 계획을 보이고, frontend 를 다시 돌려야 하면 허락을 받습니다
  help                  이 도움말

옵션 (환경변수도 그대로 씁니다)
  --dry-run        계획만 보이고 아무것도 실행하지 않습니다        (DRY_RUN=1)
  --gpu            GPU 단계를 묻지 않고 진행합니다                 (GPU=1)
  --force          최신이어도 다시 만듭니다 (엔진 제외)            (FORCE=1)
  --accept-stale   오래된 frontend 출력으로 채점해도 통과시킵니다  (FB_ACCEPT_STALE=1)
  -h, --help       이 도움말

환경변수
  MERIDIAN_WS  워크스페이스 루트          FB_DATA   uHumans2 언팩본 폴더
  FB_RUNS      결과 폴더                  FB_GT     GT tracklet h5 폴더
  FB_LOGS      로그 폴더                  FB_MODELS 엔진(.plan) 폴더
  FB_AUTHOR    status.md 작성자           FB_CLOSING status.md 끝 질문
  FB_STATUS_COPY  status.md 를 그대로 복사할 경로 (쉼표로 여러 개).
                  기본 결과 폴더의 전체 시퀀스 실행일 때만 복사합니다 — 샌드박스·부분 실행은 팀 공유본을 덮지 않습니다.
                  빈 값(FB_STATUS_COPY=)이면 복사하지 않습니다.
  frontend_benchmark/local.env 에 같은 이름으로 적어 두면 매번 쓰지 않아도 됩니다.
  이미 설정된 환경변수가 항상 이깁니다 (빈 값으로 두면 local.env 값도 쓰지 않습니다).

시퀀스 이름
  apartment_00h · 00h 처럼 줄여 써도 됩니다. 정확한 이름이 먼저이고, 여러 개와 맞으면 후보를 보여 주고 멈춥니다.
  쓸 수 있는 이름: apartment_00h apartment_01h apartment_02h office_00h office_06h office_12h

종료 코드
  0 성공 · 1 실패 · 2 사용법·입력 문제(모르는 명령 · 시퀀스 이름 · 누락) · 3 GPU 허락 없음 ·
  4 오래된 frontend 출력으로 채점함(--accept-stale 로 통과) · 130 사용자가 중단(Ctrl+C)

예시
  bash frontend_benchmark/eval.sh score                      # 지금 결과 보기 (캐시가 있으면 몇 초)
  bash frontend_benchmark/eval.sh score apartment_00h office_06h
  bash frontend_benchmark/eval.sh viewer                     # 뷰어 만들고 목록 페이지 열기
  bash frontend_benchmark/eval.sh compare latest             # 직전 결과와 비교
  GPU=1 bash frontend_benchmark/eval.sh all                  # frontend 부터 다시 (약 40~50분)
  bash frontend_benchmark/eval.sh --dry-run                  # 무엇을 할지 계획만
EOF
}

on_int() {                       # M8: Ctrl+C 는 traceback 없이 한 줄
  printf '\n   중단됨 — 다시 실행하면 끝난 단계부터 이어서 합니다.\n' >&2
  exit 130
}
trap on_int INT

suggest() {                      # suggest <입력> → 비슷한 명령 이름 (없으면 빈 줄)
  $PY - "$1" "${COMMANDS[@]}" <<'PYEOF' 2>/dev/null || true
import difflib, sys
w = sys.argv[1].strip().lower()
cands = sys.argv[2:]
hit = [c for c in cands if c == w] or difflib.get_close_matches(w, cands, n=2, cutoff=0.4)
print(' · '.join(hit))
PYEOF
}

die_unknown() {                  # C3: 모르는 명령이면 계획·GPU 단계 전에 멈춘다
  local q=$1 s
  s=$(suggest "$q")
  echo "알 수 없는 명령입니다: '$q'" >&2
  [[ -n $s ]] && echo "   혹시 이 명령인가요? $s" >&2
  echo "   쓸 수 있는 명령: ${COMMANDS[*]}" >&2
  echo "   시퀀스 이름으로 돌리려면: bash frontend_benchmark/eval.sh run <시퀀스>" >&2
  echo "   도움말: bash frontend_benchmark/eval.sh help" >&2
  exit 2
}

seq_hits() {                     # seq_hits <질의> → 맞는 시퀀스 이름들 (정확한 이름이 있으면 그것 하나)
  local q=${1#uHumans2_} s
  q=${q//_s1/}
  for s in "${ALL_SEQS[@]}"; do [[ ${s//_s1/} == "$q" || $s == "$1" ]] && { echo "$s"; return 0; }; done
  for s in "${ALL_SEQS[@]}"; do [[ ${s//_s1/} == *"$q"* ]] && echo "$s"; done
  return 0
}

resolve() {                      # resolve <질의> → 시퀀스 하나. 없거나 여러 개면 후보만 보이고 종료 코드 2 (M9)
  local q=$1 hits=()
  if [[ $q == -* ]]; then
    echo "모르는 옵션입니다: '$q'" >&2
    echo "   도움말: bash frontend_benchmark/eval.sh help" >&2
    return 2
  fi
  mapfile -t hits < <(seq_hits "$q")
  if [[ ${#hits[@]} -eq 1 ]]; then echo "${hits[0]}"; return 0; fi
  if [[ ${#hits[@]} -gt 1 ]]; then
    echo "시퀀스 이름 '$q' 가 여러 개와 맞습니다: $(short_seq "${hits[@]}")" >&2
    echo "   위 후보 중 하나를 그대로 써 주세요." >&2
  else
    echo "그런 시퀀스가 없습니다: '$q'" >&2
    echo "   쓸 수 있는 이름: $(short_seq "${ALL_SEQS[@]}")" >&2
  fi
  return 2
}

resolve_all() {                  # resolve_all [질의...] → 전역 SEQS (인자가 없으면 6개 전부)
  SEQS=()
  local q s
  if [[ $# -eq 0 ]]; then SEQS=("${ALL_SEQS[@]}"); return 0; fi
  for q in "$@"; do
    s=$(resolve "$q") || exit 2
    SEQS+=("$s")
  done
}

stream() {                       # M8: 표준입력에서 진행 줄만 골라 경과 시간과 함께 실시간으로
  local t0=$SECONDS line
  while IFS= read -r line; do
    case $line in
      '[report]'*|'[gt2d]'*|'[status]'*|'[viewer]'*|'[examples]'*|'[compare]'*) printf '   [%4ds] %s\n' "$((SECONDS - t0))" "$line" ;;
    esac
  done
}

ready_seqs() {                   # ready_seqs <무엇> <시퀀스...> → 전역 READY · SKIPPED. 없는 입력은 한 번만 설명합니다 (재리뷰 N4 · N6)
  local what=$1 s ok reason=""
  shift
  READY=(); SKIPPED=()
  for s in "$@"; do
    ok=1
    if [[ ! -f $OUT/uHumans2_$s/frontend_output.h5 ]]; then
      ok=0; reason=${reason:-"frontend 출력(frontend_output.h5)이 없습니다 — GPU=1 bash frontend_benchmark/eval.sh run <시퀀스>"}
    elif [[ $what == examples && ! -f $OUT/uHumans2_$s/score.json ]]; then
      ok=0; reason=${reason:-"채점 결과(score.json)가 없습니다 — 먼저 채점해 주세요: bash frontend_benchmark/eval.sh score"}
    fi
    if [[ $ok == 1 ]]; then READY+=("$s"); else SKIPPED+=("$s"); fi
  done
  if [[ ${#SKIPPED[@]} -gt 0 ]]; then
    echo "   건너뜁니다 (${#SKIPPED[@]}개): $(short_seq "${SKIPPED[@]}")" >&2
    echo "   이유: $reason" >&2
    echo "   결과 폴더: $OUT" >&2
  fi
  [[ ${#READY[@]} -gt 0 ]]
}

stale() {                        # stale <산출물> [입력...] → 0(참) 이면 다시 만들어야 한다
  local t=$1 d
  shift
  [[ -e $t ]] || return 0
  for d in "$@"; do
    if [[ -e $d && $d -nt $t ]]; then printf '     (입력이 산출물보다 새로움: %s)\n' "$d"; return 0; fi
  done
  return 1
}

step() {                         # step <산출물> <설명> [입력...] -- <명령...>
  local target=$1 desc=$2
  shift 2
  local deps=()
  while [[ $# -gt 0 && $1 != -- ]]; do deps+=("$1"); shift; done
  [[ ${1:-} == -- ]] && shift
  if [[ $FORCE -ne 1 ]] && ! stale "$target" "${deps[@]}"; then skip "$desc"; return 0; fi
  local t0=$SECONDS
  "$@" || { echo "   실패: $desc — 로그를 확인해 주세요" >&2; exit 1; }
  ok "$desc ($((SECONDS - t0))s)"
}

align_gate() {                   # align_gate <alignment_check.json> <frontend_output.h5> — 지금 h5 를 검증한 통과 결과만 믿는다
  $PY - "$1" "$2" <<'PYEOF'
import hashlib, json, os, sys
aj, h5 = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(aj))
except (OSError, ValueError) as e:
    print(f'     정렬 결과를 읽지 못했습니다: {e}'); sys.exit(1)
rec, st = d.get('run_h5') or {}, os.stat(h5)
if not rec:
    print('     옛 형식 alignment_check.json (검증한 h5 기록 없음) — 다시 검증해야 합니다 (파일을 지우거나 --force)'); sys.exit(1)
if not (rec.get('size') == st.st_size and rec.get('mtime_ns') == st.st_mtime_ns):
    if rec.get('size') != st.st_size:
        print(f'     다른 frontend_output.h5 의 정렬 결과입니다 (크기 {rec.get("size")} ≠ {st.st_size})'); sys.exit(1)
    h = hashlib.sha1()
    with open(h5, 'rb') as fh:
        for c in iter(lambda: fh.read(1 << 20), b''):
            h.update(c)
    if h.hexdigest() != rec.get('sha1'):
        print('     다른 frontend_output.h5 의 정렬 결과입니다 (sha1 다름)'); sys.exit(1)
ia, fs, g = d.get('input_alignment_cm') or {}, d.get('frame_shift') or {}, d.get('gate') or {}
w = lambda c: (fs.get(c) or {}).get('within_1mm')
print(f'     입력 정렬 p50 {ia.get("p50")}cm · p99 {ia.get("p99")}cm · 프레임 밀림 검사(1mm 이내 비율 −1/0/+1) {w("-1")} / {w("0")} / {w("1")}'
      f' → {"통과" if g.get("pass") else "실패"}', flush=True)
sys.exit(0 if g.get('pass') else 1)
PYEOF
}

frontend_prov() {                # frontend_prov check|record <시퀀스> — 출처 확인(0 최신 · 1 다시 필요, 이유 한 줄 출력) / 기록
  $PY "$HERE/frontend_provenance.py" "$1" --run "$OUT/uHumans2_$2" --seq "$DATA/uHumans2_$2" --code "$HERE/run_frontend.py" \
    --src "$FRONTEND_SRC" "$FRONTEND_MSGS" --engines "$MODELS/frontend_rtx3060/fastsam.plan" "$MODELS/frontend_rtx3060/clip_image.plan" \
    --cache "$OUT/.cache/sha1.json"
}

plan_gpu() {                     # plan_gpu <시퀀스...> — GPU 단계 계획을 보이고 전역 NEED_GPU 에 다시 돌릴 것을 담는다
  NEED_GPU=()
  local s why
  say "계획 — GPU 단계"
  if [[ ! -e $MODELS/frontend_rtx3060/clip_image.plan || ! -e $MODELS/frontend_rtx3060/fastsam.plan ]]; then
    echo "   · TensorRT 엔진 없음 → 빌드 필요 (약 3.5분)"; NEED_GPU+=(engines)
  fi
  for s in "$@"; do
    if [[ $FORCE -eq 1 ]]; then why="FORCE=1"
    elif why=$(frontend_prov check "$s" 2>&1); then echo "   · $(short_seq "$s"): frontend 최신 — 다시 돌리지 않습니다"; continue
    fi
    echo "   · $(short_seq "$s"): frontend 다시 필요 — $why"; NEED_GPU+=("$s")
  done
  [[ ${#NEED_GPU[@]} -eq 0 ]] && echo "   GPU 단계 없음. CPU 단계(정렬 검증 · 가시성 · 채점 · 뷰어)는 바뀐 것만 다시 합니다."
  return 0
}

gpu_ok() {                       # NEED_GPU 가 비었거나, GPU=1 이거나, 터미널에서 y 면 0
  [[ ${#NEED_GPU[@]} -eq 0 || $GPU == 1 || $GPU_APPROVED == 1 ]] && { GPU_APPROVED=1; return 0; }
  if [[ $DRY_RUN != 1 && -t 0 && -t 1 ]]; then
    local a
    read -r -p "   GPU 로 위 ${#NEED_GPU[@]}개를 다시 돌립니다 (frontend 는 시퀀스당 1~4분). 진행할까요? [y/N] " a
    [[ $a == y || $a == Y ]] && { GPU_APPROVED=1; return 0; }
  fi
  return 1
}

gpu_stop() {                     # gpu_stop [인자] — GPU 단계를 허락받지 못함: 안내하고 종료 코드 3
  echo "   GPU 단계를 실행하지 않고 멈췄습니다." >&2
  echo "   채점만 하려면 (GPU 없이, 기존 frontend 출력으로): bash frontend_benchmark/eval.sh score" >&2
  echo "   GPU 로 진행하려면:                                GPU=1 bash frontend_benchmark/eval.sh ${1:-all}" >&2
  exit 3
}

engines() {                      # --force 로는 다시 만들지 않는다 (3분). 다시 만들려면 eval.sh engines
  say "엔진 (이 PC 용, 1회)"
  local keep=$FORCE
  FORCE=${1:-0}
  step "$MODELS/frontend_rtx3060/clip_image.plan" "TensorRT 엔진 빌드" -- \
    bash -c "$PY -u $HERE/build_engines.py > $LOG/build_engines.log 2>&1"
  FORCE=$keep
}

run_seq() {
  local s=$1
  local d=$OUT/uHumans2_$s gt=$GTDIR/uHumans2_$s.h5 sd=$DATA/uHumans2_$s
  local h5=$OUT/uHumans2_$s/frontend_output.h5 aj=$OUT/uHumans2_$s/alignment_check.json vis=$OUT/gt_vis/uHumans2_$s.npz
  local why t0
  if [[ ! -d $sd ]]; then
    echo "   데이터셋이 없습니다: $sd — FB_DATA 를 확인해 주세요 (지금 $DATA)" >&2
    exit 2
  fi
  if [[ ! -f $gt ]]; then
    echo "   GT h5 가 없습니다: $gt — FB_GT 를 확인해 주세요 (지금 $GTDIR)" >&2
    exit 2
  fi
  if [[ $FORCE -ne 1 ]] && why=$(frontend_prov check "$s" 2>&1); then
    printf '   · frontend 실행 %s (최신 — 출처 기록이 지금 코드·엔진·데이터와 같습니다)\n' "$(short_seq "$s")"
  else
    [[ $FORCE -eq 1 ]] && why="FORCE=1"
    echo "   frontend 다시 필요 $(short_seq "$s") — $why"
    NEED_GPU=("$s"); gpu_ok || gpu_stop "run $s"
    t0=$SECONDS
    mkdir -p "$d"; rm -f "$h5"
    $PY -u "$HERE/run_frontend.py" --seq "$sd" --out "$d" > "$LOG/run_$s.log" 2>&1 || { echo "   실패: frontend 실행 $s — 로그를 확인해 주세요 ($LOG/run_$s.log)" >&2; exit 1; }
    frontend_prov record "$s" > /dev/null
    ok "frontend 실행 $(short_seq "$s") ($((SECONDS - t0))s) · 출처 기록"
  fi
  step "$aj" "입력 정렬 검증 $(short_seq "$s")" "$h5" "$HERE/check_alignment.py" "$gt" -- \
    bash -c "$PY $HERE/check_alignment.py --run $d --seq $sd --gt $gt --n-kf 30 > $LOG/align_$s.log 2>&1"
  align_gate "$aj" "$h5" || { echo "   입력 정렬 실패 — $aj (로그 $LOG/align_$s.log)" >&2; exit 1; }
  step "$vis" "GT 가시성 $(short_seq "$s")" "$HERE/gt_visibility.py" "$gt" -- \
    bash -c "$PY -u $HERE/gt_visibility.py --seq $sd --gt $gt --out $vis --workers 8 > $LOG/vis_$s.log 2>&1"
}

score_all() {                    # score_all [시퀀스...] — 없으면 6개 전부. 진행은 실시간(M8), 원문 로그는 <LOGS>/report.log
  say "채점 ($([[ $# -gt 0 ]] && short_seq "$@" || echo "시퀀스 6개") → summary.md)"
  local t0=$SECONDS rc=0
  local args=()
  [[ $# -gt 0 ]] && args=(--seqs "$@")
  [[ ${ACCEPT_STALE:-0} == 1 ]] && args+=(--accept-stale)
  set +e
  $PY -u "$HERE/report.py" "${args[@]}" 2>&1 | tee "$LOG/report.log" | stream
  rc=${PIPESTATUS[0]}
  set -e
  case $rc in
    0) ok "채점 완료 ($((SECONDS - t0))s)" ;;
    130)
      echo "   중단됨 — 다시 실행하면 끝난 단계부터 이어서 합니다." >&2
      exit 130 ;;
    2)
      echo "   채점하지 못한 시퀀스가 있습니다 (종료 코드 2). 위 '누락' 줄에 없는 파일과 다음에 할 일이 적혀 있습니다." >&2
      echo "   경로 설정을 확인해 주세요: FB_DATA=$DATA · FB_GT=$GTDIR · FB_RUNS=$OUT" >&2
      echo "   전체 로그: $LOG/report.log" >&2
      exit 2 ;;
    4)
      echo "   오래된 frontend 출력으로 채점했습니다 (종료 코드 4). 위 '경고' 줄이 어느 시퀀스인지 알려 줍니다." >&2
      $PY "$HERE/status_md.py" | sed 's/^/   /'      # summary.md 와 status.md 가 같은 경고를 담게 (재리뷰 N3)
      echo "   summary.md · status.md 머리에 같은 경고와 기준 시각을 적었습니다." >&2
      echo "   지금 코드로 다시 돌리려면: GPU=1 bash frontend_benchmark/eval.sh all" >&2
      echo "   이대로 받아들이려면:       bash frontend_benchmark/eval.sh score --accept-stale" >&2
      exit 4 ;;
    *)
      tail -25 "$LOG/report.log" >&2
      echo "   채점에 실패했습니다 (종료 코드 $rc) — 전체 로그 $LOG/report.log" >&2
      exit "$rc" ;;
  esac
  $PY "$HERE/status_md.py" | sed 's/^/   /'
  if [[ -f $OUT/viewer/index.html ]]; then   # 뷰어 목록 페이지의 '오래됨' 표시만 1초 안에 갱신 (⑦ build_viewer --index-only)
    $PY "$HERE/build_viewer.py" --index-only 2>&1 | sed 's/^/   /' || echo "   뷰어 목록 페이지는 갱신하지 못했습니다 — bash frontend_benchmark/eval.sh viewer" >&2
  fi
  show
}

viewer_build() {                 # viewer_build [시퀀스...]
  local rc=0
  set +e
  $PY -u "$HERE/build_viewer.py" "$@" 2>&1 | tee "$LOG/viewer.log" | stream
  rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]]; then
    tail -20 "$LOG/viewer.log" >&2
    echo "   뷰어를 만들지 못했습니다 (종료 코드 $rc) — 전체 로그 $LOG/viewer.log" >&2
    echo "   채점 결과가 없으면 먼저 실행해 주세요: bash frontend_benchmark/eval.sh score" >&2
    exit "$rc"
  fi
}

show() {
  if [[ ! -f $OUT/summary.md ]]; then
    echo "아직 채점 결과가 없습니다 ($OUT/summary.md 가 없습니다)." >&2
    echo "먼저 채점해 주세요: bash frontend_benchmark/eval.sh score" >&2
    exit 2
  fi
  say "채점표: $OUT/summary.md"
  sed -n '/^## 1\./,/^## 2\./p' "$OUT/summary.md" | head -n -1
  printf '\n   전체 표: %s\n   팀 공유용 요약: %s\n   이전과 비교: bash frontend_benchmark/eval.sh compare latest\n' \
    "$OUT/summary.md" "$OUT/status.md"
  printf '   시퀀스별 상세: %s/uHumans2_<시퀀스>/score_{episodes,observations}.csv\n' "$OUT"
}

cmd_all() {                      # 전부: GPU 계획 → (DRY_RUN 이면 끝) → 허락 → 엔진 · 시퀀스 6개 · 채점 · 뷰어
  plan_gpu "${ALL_SEQS[@]}"
  if [[ $DRY_RUN == 1 ]]; then echo "   DRY_RUN=1 — 계획만 보이고 아무것도 실행하지 않았습니다."; exit 0; fi
  gpu_ok || gpu_stop
  engines
  say "시퀀스 6개"
  local s
  for s in "${ALL_SEQS[@]}"; do run_seq "$s"; done
  score_all
  say "뷰어"; viewer_build
}

cmd_run() {                      # cmd_run <시퀀스...>: 그 시퀀스만 실행 · 정렬 검증 · 가시성 후 전체 채점 · 그 시퀀스 뷰어
  local seqs=("$@") s
  plan_gpu "${seqs[@]}"
  if [[ $DRY_RUN == 1 ]]; then echo "   DRY_RUN=1 — 계획만 보이고 아무것도 실행하지 않았습니다."; exit 0; fi
  gpu_ok || gpu_stop "run ${seqs[*]}"
  engines
  for s in "${seqs[@]}"; do
    say "시퀀스 $(short_seq "$s")"
    run_seq "$s"
  done
  score_all
  say "뷰어 $(short_seq "${seqs[@]}")"; viewer_build "${seqs[@]}"
}

cmd_examples() {                 # cmd_examples <시퀀스...>
  local s rc=0 extra=()
  [[ $FORCE -eq 1 ]] && extra=(--force)
  for s in "$@"; do
    say "판정 그림 $(short_seq "$s")"
    $PY "$HERE/render_examples.py" --run "$OUT/uHumans2_$s" --seq "$DATA/uHumans2_$s" \
      --gt "$GTDIR/uHumans2_$s.h5" --vis "$OUT/gt_vis/uHumans2_$s.npz" --out "$OUT/uHumans2_$s/examples" "${extra[@]}" || rc=$?
  done
  if [[ $rc -ne 0 ]]; then
    echo "   판정 그림을 만들지 못한 시퀀스가 있습니다 (종료 코드 $rc) — 위 줄에 없는 파일과 다음에 할 일이 적혀 있습니다." >&2
    exit "$rc"
  fi
}

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then return 0; fi   # source 로 읽으면 함수만 정의하고 끝

# ---------------------------------------------------------------- 옵션 · 명령 (M9 · C3)
while [[ $# -gt 0 ]]; do         # 명령 앞의 옵션
  case $1 in
    --dry-run) DRY_RUN=1 ;;
    --gpu) GPU=1 ;;
    --force) FORCE=1 ;;
    --accept-stale) ACCEPT_STALE=1 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    -*) echo "모르는 옵션입니다: '$1'" >&2; echo "   도움말: bash frontend_benchmark/eval.sh help" >&2; exit 2 ;;
    *) break ;;
  esac
  shift
done
CMD=all
if [[ $# -gt 0 ]]; then CMD=$1; shift; fi
REST=()
while [[ $# -gt 0 ]]; do         # 명령 뒤의 인자 — 아는 전역 옵션은 여기서, 나머지는 명령에 넘긴다 (예: test --only)
  case $1 in
    --dry-run) DRY_RUN=1 ;;
    --gpu) GPU=1 ;;
    --force) FORCE=1 ;;
    --accept-stale) ACCEPT_STALE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) REST+=("$1") ;;
  esac
  shift
done

case $CMD in
  all)
    need_dirs
    [[ ${#REST[@]} -eq 0 ]] || { resolve_all "${REST[@]}"; cmd_run "${SEQS[@]}"; exit 0; }
    cmd_all ;;
  run)
    [[ ${#REST[@]} -gt 0 ]] || { echo "어떤 시퀀스를 돌릴지 알려 주세요: bash frontend_benchmark/eval.sh run office_06h" >&2; exit 2; }
    resolve_all "${REST[@]}"
    need_dirs
    cmd_run "${SEQS[@]}" ;;
  score)
    resolve_all "${REST[@]}"
    need_dirs
    if [[ ${#REST[@]} -eq 0 ]]; then score_all; else score_all "${SEQS[@]}"; fi ;;
  compare) $PY "$HERE/compare.py" "${REST[@]}" ;;
  history) $PY "$HERE/compare.py" --list ;;
  status) $PY "$HERE/status_md.py" "${REST[@]}" ;;
  show)  show ;;
  engines) need_dirs; engines 1 ;;
  test)
    need_dirs
    say "채점기 자체 검증 (frontend_benchmark/tests/test_*.py)"
    targs=(); only=()
    for a in ${REST[@]+"${REST[@]}"}; do          # 파일 이름을 그냥 줘도 되게 (tests/test_cli.py · test_cli.py 둘 다)
      case $a in
        -*) targs+=("$a") ;;
        *test_*.py) only+=("$(basename "$a")") ;;
        *) targs+=("$a") ;;
      esac
    done
    [[ ${#only[@]} -gt 0 ]] && targs+=(--only "${only[@]}")
    trc=0
    $PY -u "$HERE/run_tests.py" ${targs[@]+"${targs[@]}"} || trc=$?
    if [[ ${#REST[@]} -eq 0 && -f $OUT/summary.json ]]; then   # 전부 돌렸을 때만 status.md 의 '자체 검증 N항목' 을 갱신 (재리뷰 N2)
      $PY "$HERE/status_md.py" | sed 's/^/   /'
    elif [[ ${#REST[@]} -gt 0 ]]; then
      echo "   (일부만 돌렸으므로 status.md 의 자체 검증 항목 수는 그대로 둡니다 — 전부 돌리려면 인자 없이 실행해 주세요)"
    fi
    exit $trc ;;
  viewer)
    resolve_all "${REST[@]}"
    need_dirs
    say "뷰어"
    if ! ready_seqs viewer "${SEQS[@]}"; then
      echo "   뷰어를 만들 수 있는 시퀀스가 없습니다 (종료 코드 2)." >&2
      echo "   채점부터 하려면: bash frontend_benchmark/eval.sh score" >&2
      exit 2
    fi
    viewer_build "${READY[@]}"
    printf '\n   목록 페이지(여기서 시작): %s\n   시퀀스 하나: %s\n   브라우저로 바로 열 수 있습니다 (서버가 필요 없습니다).\n' \
      "$OUT/viewer/index.html" "$OUT/viewer/uHumans2_<시퀀스>/index.html" ;;
  examples)
    resolve_all "${REST[@]}"
    need_dirs
    if ! ready_seqs examples "${SEQS[@]}"; then
      echo "   판정 그림을 만들 수 있는 시퀀스가 없습니다 (종료 코드 2)." >&2
      echo "   채점부터 하려면: bash frontend_benchmark/eval.sh score" >&2
      exit 2
    fi
    cmd_examples "${READY[@]}" ;;
  help) usage ;;
  *)
    if hits=$(seq_hits "$CMD") && [[ -n $hits ]]; then
      resolve_all "$CMD" "${REST[@]}"
      cmd_run "${SEQS[@]}"
    else
      die_unknown "$CMD"
    fi ;;
esac
