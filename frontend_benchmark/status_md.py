#!/usr/bin/env python3
"""팀 공유용 짧은 진행 상황 md — <RUNS>/summary.json · <RUNS>/tests_summary.json → <RUNS>/status.md.

모양 (사용자 요청 "너무 난잡해 핵심만" · 승인한 짧은 형식, 수정 ⑤ 후속 F3 · 사용성 리뷰 m5)
  # 제목
  <날짜> · <작성자>      날짜 = summary.json 의 채점 시각 · 작성자 = --author → FB_AUTHOR → local.env → git config user.name → 없음
  (오래된 frontend 출력이면 경고 한 줄)
  소개 2줄
  ## 결과  표  시퀀스 | 트랙 재현율 (Easy) | F@20cm | PQ | IDF1 | HOTA_α
         열 이름 · 자리수 · 단위는 terms.py METRICS 한 곳에서 온다 (summary.md · 뷰어와 같은 글자)
         트랙 재현율 (Easy) = level_3d.easy 의 n_detected / n (개수에서 한 번만 반올림, half-up) · 그 밖은 JSON 값을 한 번 반올림
  글머리표 4개, 한 줄씩 120자 이하, 괄호 안에 괄호 없음
    **위치** · **분할** · **추적**: 핵심 숫자(표 칸의 최소~최대) + 짧은 판정 한 마디 (아래 JUDGE)
    용어 한 줄: 트랙 재현율 · Easy · HOTA_α (점수 열은 1이 만점)
  ## 뷰어 한 줄 (RUNS 기준 실제 경로) · ## 한계 한 줄(자체 검증 항목 수) · 끝 질문(설정에 있을 때만)
설정 (m5) 하드코딩하지 않는다. 우선순위는 명령줄 옵션 → 환경변수 → local.env → 기본값이다.
  FB_AUTHOR   작성자 (없으면 git config user.name, 그것도 없으면 이름 줄을 빼고 날짜만)
  FB_CLOSING  끝 질문 문구 (비어 있으면 질문을 넣지 않는다). 이 워크스페이스는 local.env 에서 "이대로 진행할까요?" 로 켜 둔다
  FB_STATUS_COPY  만든 status.md 를 그대로 복사할 경로 (쉼표로 여러 개). 손으로 복사하지 않게 — 이 워크스페이스는 Desktop · docs 사본
                  사본은 **기본 결과 폴더(FB_RUNS 를 바꾸지 않음)의 전체 시퀀스 실행**일 때만 만든다 (재리뷰 N1):
                  샌드박스 실행이나 부분 실행(--seqs)이 팀 공유본을 덮어쓰면 안 되기 때문이다. 건너뛰면 이유를 화면에 적는다.
                  값이 빈 문자열이면(FB_STATUS_COPY= ) 사본을 만들지 않는다. 명령줄 --copy-to 는 사람이 직접 고른 것이라 조건 없이 따른다.
  local.env   frontend_benchmark/local.env (KEY=VALUE, git 에 올리지 않는다 — .gitignore)
판정 문구 기준 JUDGE (우리가 정한 읽기용 기준 — 표준 등급이 아니다. docs/METRICS.md 에 같은 표)
  좋다고 말하려면 가장 나쁜 시퀀스도 기준을 넘어야 한다: 위치 = F@20cm 최솟값, 분할 = PQ 최솟값, 추적 = GT 트랙당 예측 ID 수 최댓값.
  값은 표·글머리표에 보이는 반올림된 숫자로 판정한다 (읽는 사람이 본 숫자와 판정이 어긋나지 않게).
  위치  F@20cm ≥ 0.90 잘 겹침 · ≥ 0.75 대체로 겹침 · 그 밖 많이 어긋남
  분할  PQ ≥ 0.70 좋음 · ≥ 0.50 보통 · 그 밖 약함
  추적  ID 수 ≤ 1.2 거의 유지 · ≤ 1.5 가끔 바뀜 · 그 밖 자주 바뀜
용어  감사 D 결정: '3D 재현율' 을 쓰지 않고 '트랙 재현율', ID 수는 'GT 트랙당 예측 ID'. F@20cm 를 '20cm 이내 비율'로 설명하지 않는다.
      과다분할 · 과소분할로 쓴다 — 같은 뜻의 다른 낱말을 섞지 않는다 (표시 이름은 terms.py 한 곳).
검증: test_status_md.py 가 만든 md 를 다시 읽어 모든 칸 · 숫자 · 형식을 summary.json 과 따로 계산해 대조한다
      (python test_status_md.py --verify <RUNS>/status.md <RUNS>/summary.json <RUNS>/tests_summary.json).
"""
import argparse
import json
import os
import subprocess
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402
import provenance as PV  # noqa: E402
import terms as T  # noqa: E402

HERE = Path(__file__).resolve().parent
COLS = ['시퀀스', T.metric_label('track_recall_easy'), T.metric_label('f20', population=False),
        T.metric_label('pq', population=False), T.metric_label('idf1', population=False),
        T.metric_label('hota_alpha', population=False)]


# ---------------------------------------------------------------- 설정 (m5)
def local_env(path=None):
    """frontend_benchmark/local.env (KEY=VALUE) → dict. 없으면 빈 dict."""
    p = Path(path) if path else HERE / 'local.env'
    out = {}
    try:
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


def option(name, default=''):
    """환경변수 → local.env → 기본값."""
    v = os.environ.get(name)
    if v is None:
        v = local_env().get(name)
    return default if v is None else v


def git_user_name():
    try:
        r = subprocess.run(['git', 'config', 'user.name'], capture_output=True, text=True, timeout=5, cwd=str(HERE))
        return r.stdout.strip() if r.returncode == 0 else ''
    except (OSError, subprocess.SubprocessError):
        return ''


def author_name(explicit=None):
    """--author 가 있으면 그것 (빈 문자열 = 이름 줄 빼기), 없으면 FB_AUTHOR → local.env → git config user.name."""
    if explicit is not None:
        return explicit
    return option('FB_AUTHOR') or git_user_name()


def closing_question(explicit=None):
    return explicit if explicit is not None else option('FB_CLOSING')


def g(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or d.get(k) is None:
            return None
        d = d[k]
    return d


def dec(x, nd):
    return None if x is None else Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), ROUND_HALF_UP)


def short(s):
    return T.seq_label(s)


def idf1(m):
    if m and all(k in m for k in ('IDTP', 'IDFP', 'IDFN')) and (2 * m['IDTP'] + m['IDFP'] + m['IDFN']):
        return T.fmt('idf1', Decimal(2 * m['IDTP']) / Decimal(2 * m['IDTP'] + m['IDFP'] + m['IDFN']))
    return T.fmt('idf1', g(m, 'IDF1'))


def cells(r):
    """표 한 줄의 값 칸 — 형식은 terms.py METRICS (summary.md · 뷰어와 같은 글자)."""
    lv = g(r, 'level_3d', 'easy') or {}
    return [T.fmt_kn('track_recall_easy', lv.get('n_detected'), lv.get('n')),
            T.fmt('f20', g(r, 'geometry', 'summary', 'F@20', 'median')),
            T.fmt('pq', g(r, 'seg2d', 'all', 'PQ')),
            idf1(g(r, 'mot', 'metrics')),
            T.fmt('hota_alpha', g(r, 'mot', 'metrics', 'HOTA'))]


def span(vals, unit=''):
    v = [x for x in vals if x not in (None, '–')]
    if not v:
        return '–'
    key = lambda x: Decimal(str(x).rstrip('%'))
    lo, hi = min(v, key=key), max(v, key=key)
    return f'{lo}{unit}' if lo == hi else f'{lo}~{hi}{unit}'


# 판정 문구 — (기준, 문구). position · segmentation 은 값 ≥ 기준인 첫 줄, tracking 은 값 ≤ 기준인 첫 줄 (모듈 docstring)
JUDGE = {
    'position': ((0.90, '표면이 GT 와 잘 겹칩니다'), (0.75, '표면이 GT 와 대체로 겹칩니다'), (float('-inf'), '표면이 GT 와 많이 어긋납니다')),
    'segmentation': ((0.70, '분할이 좋습니다'), (0.50, '분할은 보통입니다'), (float('-inf'), '분할은 아직 약합니다')),
    'tracking': ((1.2, '보이는 동안 ID 가 거의 유지됩니다'), (1.5, '보이는 동안에도 ID 가 가끔 바뀝니다'),
                 (float('inf'), '보이는 동안에도 ID 가 자주 바뀝니다')),
}
JUDGE_RULE = {                                  # 문서·summary.md 가 같이 쓰는 설명 (기준 = 가장 나쁜 시퀀스)
    'position': 'F@20cm 최솟값 ≥ 0.90 잘 겹침 · ≥ 0.75 대체로 겹침 · 그 밖 많이 어긋남',
    'segmentation': 'PQ 최솟값 ≥ 0.70 좋음 · ≥ 0.50 보통 · 그 밖 약함',
    'tracking': 'GT 트랙당 예측 ID 수 최댓값 ≤ 1.2 거의 유지 · ≤ 1.5 가끔 바뀜 · 그 밖 자주 바뀜',
}


def judge(kind, value):
    rules = JUDGE[kind]
    v = float(value)
    if kind == 'tracking':
        return next(t for th, t in rules if v <= th + 1e-12)
    return next(t for th, t in rules if v >= th - 1e-12)


def ids_values(rows):
    return [str(dec(g(r, 'mot', 'metrics', 'mean_ids_per_track'), T.metric('ids_per_track')['decimals']))
            for _, r in rows if g(r, 'mot', 'metrics', 'mean_ids_per_track') is not None]


def verdict_bullets(rows):
    """위치 · 분할 · 추적 글머리표 3줄 — summary.md 의 '한눈에' 와 status.md 가 같은 문장을 쓴다."""
    rows = list(rows.items()) if isinstance(rows, dict) else list(rows)
    C = {s: cells(r) for s, r in rows}
    num = lambda i: [C[s][i] for s, _ in rows if C[s][i] not in (None, '–')]
    ids_v = ids_values(rows)
    st_sum = lambda k: sum(int(g(r, 'seg2d', 'all', 'gt_status', k) or 0) for _, r in rows)
    pick = lambda kind, vals, worst: judge(kind, worst(Decimal(str(x)) for x in vals)) if vals else '–'
    return [f'- **위치**: F@20cm {span(num(1))} — {pick("position", num(1), min)}.',
            f'- **분할**: PQ {span(num(2))} · 과다분할 {st_sum("split")}건 · 과소분할 {st_sum("merged")}건 — '
            f'{pick("segmentation", num(2), min)}.',
            f'- **추적**: IDF1 {span(num(3))} · GT 트랙당 예측 ID {span(ids_v)}개 — {pick("tracking", ids_v, max)}.']


def scored_at(summary):
    """이 표가 무슨 시각의 채점 결과인지 (가장 최근에 끝난 단계)."""
    return max((st.get('finished') or '' for r in summary.values() for st in (r.get('report_steps') or {}).values()), default='')


def stale_note(summary):
    """C1 · N3 — 오래된 frontend 출력으로 채점한 시퀀스가 있으면 경고 한 줄(기준 시각 포함), 없으면 None.
    summary.md 의 경고와 같은 근거(summary.json 의 frontend.provenance_check)에서 나온다."""
    bad = {s: g(r, 'frontend', 'provenance_check', 'reasons') or []
           for s, r in summary.items() if g(r, 'frontend', 'provenance_check', 'status') == 'stale'}
    if not bad:
        return None
    who = ', '.join(short(s) for s in bad)
    why = next((r[0] for r in bad.values() if r), '출처 기록이 지금과 다름')
    when = scored_at(summary)
    return (f'> ⚠ 주의: {who} 는 지금 frontend 코드보다 오래된 출력으로 채점했습니다 — {why}. '
            f'이 표는 {when or "시각 불명"} 채점 결과입니다. 다시 돌리려면 `GPU=1 bash frontend_benchmark/eval.sh all` 을 실행해 주세요.')


def default_runs():
    """FB_RUNS 를 주지 않았을 때의 결과 폴더 (paths.py 기본값)."""
    return Path(P.WS) / 'runs/frontend_eval'


def copy_targets(summary, runs=None, value=None, explicit=False):
    """N1 — status.md 사본을 어디에 쓸지 → (경로 목록, 건너뛴 이유 또는 '').
    설정(FB_STATUS_COPY)으로 온 값은 '기본 결과 폴더 + 전체 시퀀스' 일 때만 쓴다. --copy-to 는 조건 없이 쓴다."""
    paths_ = [Path(x.strip()).expanduser() for x in (value or '').split(',') if x.strip()]
    if not paths_:
        return [], ''
    if explicit:
        return paths_, ''
    runs = Path(runs or P.RUNS)
    if runs.resolve() != default_runs().resolve():
        return [], f'결과 폴더가 기본값이 아니라 사본을 만들지 않았습니다 ({runs})'
    missing = [s for s in P.SEQUENCES if s not in summary]
    if missing:
        return [], f'시퀀스 {len(summary)}/{len(P.SEQUENCES)}개만 있는 결과라 사본을 만들지 않았습니다 (빠짐: {", ".join(short(s) for s in missing[:3])})'
    return paths_, ''


def render_status(summary, tests, author=None, closing=None):
    rows = list(summary.items())
    C = {s: cells(r) for s, r in rows}
    first = rows[0][1] if rows else {}
    date = max((st.get('finished') or '' for _, r in rows for st in (r.get('report_steps') or {}).values()), default='')[:10]
    who = author_name(author)
    area = g(first, 'visibility', 'area_min_px')
    crit = g(first, 'level_3d_criteria', 'easy') or {}
    alpha = g(first, 'mot', 'params', 'min_frac')
    runs_rel = os.path.relpath(P.RUNS, P.WS) if str(P.RUNS).startswith(str(P.WS)) else str(P.RUNS)
    L = ['# Frontend 평가 진행 상황', '',
         f'{date or "날짜 없음"} · {who}' if who else (date or '날짜 없음'), '',
         '<!-- status_md.py 가 summary.json · tests_summary.json 에서 자동 생성. 판정 문구 기준은 status_md.JUDGE -->', '']
    note = stale_note(summary)
    if note:
        L += [note, '']
    L += [f'uHumans2 {len(rows)}개 시퀀스에 frontend(SAM · GeoBuilder · CLIP · Tracker)를 돌리고 GT와 대조했습니다.',
          '카메라 위치는 GT 기록을 써서 SLAM 오차 없이 frontend만 평가했습니다.', '',
          '## 결과', '',
          '| ' + ' | '.join(COLS) + ' |', '|---|' + '---:|' * (len(COLS) - 1)]
    for s, _ in rows:
        L.append(f'| {short(s)} | ' + ' | '.join(C[s]) + ' |')
    easy = ', '.join(x for x in (f'폭 {crit["min_dim_gt"]}px 초과' if 'min_dim_gt' in crit else None,
                                 f'잘림 {int(round(crit["trunc_le"] * 100))}% 이하' if 'trunc_le' in crit else None,
                                 f'가림 {int(round(crit["occ_le"] * 100))}% 이하' if 'occ_le' in crit else None) if x)
    L += [''] + verdict_bullets(rows) + [
        f'- 용어: 트랙 재현율 = {area if area is not None else "?"}px 이상 보인 물체를 한 번이라도 잡은 비율 · '
        f'Easy = {easy or "gt_difficulty_2d 기준"} · HOTA_α = α {alpha if alpha is not None else "?"} 한 점 · 점수는 1이 만점']
    t = (tests or {}).get('totals') or {}
    L += ['', '## 뷰어', '',
          f'`{runs_rel}/viewer/index.html` 을 브라우저로 엽니다. 맵에서 검출·미검출을 색으로 보고, 물체를 누르면 이유와 해당 이미지가 나옵니다.',
          '', '## 한계', '']
    if t:
        extra = (f', 실패 {t.get("failed")}개' if t.get('failed') else '') + (f', 건너뜀 {t.get("skipped")}개' if t.get('skipped') else '')
        verdict = f'자체 검증 {t.get("n_ok")}항목' + (f'({extra[2:]})' if extra else '') + '과 뷰어 확인은 했지만'
    else:
        verdict = '자체 검증 결과가 없어서'
    L += [f'채점기를 AI로 작성했습니다. {verdict} **채점 기준이 팀 기준과 맞는지는 사람이 봐야 합니다.**']
    q = closing_question(closing)
    if q:
        L += ['', q]
    return '\n'.join(L) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--author', default=None, help='작성자 (기본: FB_AUTHOR → local.env → git config user.name)')
    ap.add_argument('--closing', default=None, help='끝 질문 문구 (기본: FB_CLOSING → local.env)')
    ap.add_argument('--no-closing', action='store_true', help='끝 질문 없이')
    ap.add_argument('--copy-to', default=None, help='만든 status.md 를 이 경로들에도 복사 (쉼표 구분, 기본 FB_STATUS_COPY)')
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    sp = P.RUNS / 'summary.json'
    if not sp.exists():
        print(f'채점 결과가 없습니다 ({sp} 없음) — 먼저 `bash frontend_benchmark/eval.sh score` 를 실행해 주세요.', file=sys.stderr)
        return 2
    tp = P.RUNS / 'tests_summary.json'
    tests = json.loads(tp.read_text()) if tp.exists() else {}
    out = P.RUNS / 'status.md'
    closing = '' if a.no_closing else a.closing
    summary = json.loads(sp.read_text())
    wrote = PV.write_text_if_changed(out, render_status(summary, tests, author=a.author, closing=closing))
    print(f'[status] {out} {"갱신" if wrote else "변경 없음"}')
    dests, why = copy_targets(summary, runs=P.RUNS, value=a.copy_to if a.copy_to is not None else option('FB_STATUS_COPY'),
                              explicit=a.copy_to is not None)
    if why:
        print(f'[status] 사본 건너뜀 — {why}')
    for dest in dests:                                         # m5: 팀 공유 사본을 손으로 복사하지 않게 (조건은 N1)
        try:
            c = PV.write_text_if_changed(dest, out.read_text())
            print(f'[status] 사본 {dest} {"갱신" if c else "변경 없음"}')
        except OSError as e:
            print(f'[status] 사본을 쓰지 못했습니다 ({dest}): {e}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
