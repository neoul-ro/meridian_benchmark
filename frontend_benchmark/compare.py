#!/usr/bin/env python3
"""이전 채점 결과와 지금 결과를 견줘 본다 (사용성 리뷰 C2) — "내가 고친 뒤 좋아졌나?" 에 답하는 표.

  python compare.py                  # 가장 최근 보관본(latest) ↔ 지금 <RUNS>/summary.json
  python compare.py 20260918-031500  # 그 보관본 ↔ 지금
  python compare.py --list           # 보관 목록 (id · 보관 시각 · 시퀀스 수 · frontend 소스 sha1)
  python compare.py --old A --new B  # 파일 두 개를 직접 (summary.json 경로)

보관은 report.py 가 채점할 때마다 한다 (<RUNS>/history/<시각>/summary.json · summary.md · status.md · provenance.json).
지표 이름 · 자리수 · '클수록 좋은지' 는 terms.py METRICS 한 곳에서 온다 — 방향에 맞춰 좋아짐 · 나빠짐을 적는다.
"""
import argparse
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402
import terms as T  # noqa: E402

# 표에 넣는 핵심 지표 (terms.py 키) → summary.json 에서 값을 꺼내는 법
ROWS = ['track_recall_easy', 'track_recall_eligible', 'f20', 'pq', 'idf1', 'hota_alpha', 'ids_per_track', 'n_split', 'n_merged']


def g(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or d.get(k) is None:
            return None
        d = d[k]
    return d


def value(key, r):
    """지표 하나의 원값 (percent 는 0~1 비율, count 는 개수). 없으면 None."""
    if key == 'track_recall_easy':
        lv = g(r, 'level_3d', 'easy') or {}
        n, k = lv.get('n'), lv.get('n_detected')
        return (k / n) if n and k is not None else g(r, 'level_3d', 'easy', 'recall')
    if key == 'track_recall_eligible':
        c = g(r, 'counts_3d', 'eligible') or {}
        if c.get('n'):
            return c['detected'] / c['n']
        return g(r, 'visibility', 'recall_detectable')
    if key == 'f20':
        return g(r, 'geometry', 'summary', 'F@20', 'median')
    if key == 'pq':
        return g(r, 'seg2d', 'all', 'PQ')
    if key == 'idf1':
        m = g(r, 'mot', 'metrics') or {}
        if all(x in m for x in ('IDTP', 'IDFP', 'IDFN')) and (2 * m['IDTP'] + m['IDFP'] + m['IDFN']):
            return 2 * m['IDTP'] / (2 * m['IDTP'] + m['IDFP'] + m['IDFN'])
        return m.get('IDF1')
    if key == 'hota_alpha':
        return g(r, 'mot', 'metrics', 'HOTA')
    if key == 'ids_per_track':
        return g(r, 'mot', 'metrics', 'mean_ids_per_track')
    if key == 'n_split':
        return g(r, 'seg2d', 'all', 'gt_status', 'split')
    if key == 'n_merged':
        return g(r, 'seg2d', 'all', 'gt_status', 'merged')
    return None


def delta_text(key, old, new):
    """'0.42 → 0.52 (+0.10 좋아짐)'. 값이 없으면 '–'."""
    if old is None and new is None:
        return '–'
    if old is None:
        return f'– → {T.fmt(key, new)} (새로 채점)'
    if new is None:
        return f'{T.fmt(key, old)} → – (이번 결과 없음)'
    m = T.metric(key)
    nd = m.get('decimals', 2)
    kind = m.get('kind')
    d = Decimal(repr(float(new))) - Decimal(repr(float(old)))
    if kind == 'percent':
        d = (d * 100).quantize(Decimal(1).scaleb(-nd), ROUND_HALF_UP)
        unit = '%p'
    elif kind == 'count':
        d = d.quantize(Decimal(1), ROUND_HALF_UP)
        unit = '건'
    else:
        d = d.quantize(Decimal(1).scaleb(-nd), ROUND_HALF_UP)
        unit = m.get('unit', '')
    if d == 0:
        return f'{T.fmt(key, old)} → {T.fmt(key, new)} (변화 없음)'
    better = (d > 0) == T.metric_higher_better(key)
    return f'{T.fmt(key, old)} → {T.fmt(key, new)} ({"+" if d > 0 else ""}{d}{unit} {"좋아짐" if better else "나빠짐"})'


def render_compare(old_rows, new_rows, old_id='', old_info=None, new_info=None):
    """시퀀스 × 핵심 지표 변화표 (한국어)."""
    old_info = old_info or {}
    seqs = [s for s in (list(new_rows) + [x for x in old_rows if x not in new_rows])]
    order = {s: i for i, s in enumerate(P.SEQUENCES)}
    seqs.sort(key=lambda s: (order.get(s, 99), s))
    when = old_info.get('archived_at') or '–'
    L = [f'# 이전과 비교 — 보관본 `{old_id}` ({when}) ↔ 지금', '']
    src_old, src_new = old_info.get('frontend_src_sha1'), (new_info or {}).get('frontend_src_sha1')
    if src_old and src_new:
        L += [('- frontend 소스: **바뀌었습니다** ' if src_old != src_new else '- frontend 소스: 같습니다 ')
              + f'(`{str(src_old)[:12]}` → `{str(src_new)[:12]}`)']
    code_old, code_new = old_info.get('benchmark_code_sha1'), (new_info or {}).get('benchmark_code_sha1')
    if code_old and code_new:
        L += [('- 채점 코드: **바뀌었습니다** ' if code_old != code_new else '- 채점 코드: 같습니다 ')
              + f'(`{str(code_old)[:12]}` → `{str(code_new)[:12]}`)']
    if src_old and src_new and src_old != src_new:
        L += ['- 채점 코드까지 바뀌었다면 숫자 차이에 두 가지가 섞여 있습니다 — 한 번에 하나만 바꾸는 편이 읽기 쉽습니다.'] \
            if code_old and code_new and code_old != code_new else []
    L += ['', '| 지표 | ' + ' | '.join(T.seq_label(s) for s in seqs) + ' |', '|---|' + '---|' * len(seqs)]
    better = worse = same = 0
    for key in ROWS:
        cells = []
        for s in seqs:
            o, n = value(key, old_rows.get(s) or {}), value(key, new_rows.get(s) or {})
            txt = delta_text(key, o, n)
            better += '좋아짐' in txt
            worse += '나빠짐' in txt
            same += '변화 없음' in txt
            cells.append(txt)
        L.append(f'| {T.metric_label(key)} | ' + ' | '.join(cells) + ' |')
    L += ['', f'좋아진 칸 {better}개 · 나빠진 칸 {worse}개 · 변화 없음 {same}개.']
    if not better and not worse:
        L += ['', '두 결과의 핵심 지표가 모두 같습니다 (변화 없음).']
    L += ['', '- 방향: 트랙 재현율 · F@20cm · PQ · IDF1 · HOTA_α 는 클수록, GT 트랙당 예측 ID 수 · 과다분할 · 과소분할 건수는 작을수록 좋습니다.',
          '- 자세한 표는 `summary.md`, 보관본은 `history/<id>/summary.md` 에 있습니다.']
    return '\n'.join(L) + '\n'


def load_rows(path, what):
    """보관본·지금 결과의 summary.json 을 읽는다 → (dict 또는 None, 문제 설명).
    옛 형식이나 깨진 파일이면 traceback 대신 한 줄로 알린다 (재리뷰 N5)."""
    try:
        d = json.loads(Path(path).read_text())
    except OSError as e:
        return None, f'{what} 을(를) 열 수 없습니다 ({path}): {e.strerror}'
    except ValueError as e:
        return None, f'{what} 이(가) 올바른 JSON 이 아닙니다 ({path}): {e}'
    if not isinstance(d, dict) or not d or not all(isinstance(v, dict) for v in d.values()):
        return None, (f'{what} 이(가) 지금 형식이 아닙니다 ({path}) — 이 명령은 "시퀀스 이름 → 채점 블록" 꼴만 읽습니다. '
                      f'옛 보관본이면 그 폴더의 summary.md 를 직접 보거나, 지금 코드로 다시 채점해 주세요.')
    return d, ''


def current_info():
    """지금 코드·frontend 소스의 출처 (보관본의 provenance.json 과 같은 키) — 표 위에 '무엇이 바뀌었나' 한 줄을 적기 위해."""
    out = {}
    try:
        import frontend_provenance as FP
        import provenance as PV
        H = PV.Hasher(P.RUNS / '.cache' / 'sha1.json')
        roots = [p for p in (P.FRONTEND_SRC, P.FRONTEND_MSGS) if Path(p).exists()]
        if roots:
            out['frontend_src_sha1'] = FP.src_digest(roots, H)[0]
        files = [P.HERE / n for n in ('report.py', 'score_frontend.py', 'score_2d.py', 'score_geometry.py', 'score_mot.py',
                                      'gt_labels_2d.py', 'gt_difficulty_2d.py', 'match3d.py', 'tide_rules.py') if (P.HERE / n).exists()]
        out['benchmark_code_sha1'] = PV.code_digest(files, [P.HERE, P.BENCH_PKG])[0]
    except Exception:  # noqa: BLE001  (비교 표는 이 줄 없이도 쓸 수 있다)
        pass
    return out


def history_list(runs=None):
    """보관된 결과 목록 (오래된 것부터) → [(id, 폴더, provenance dict)]."""
    runs = Path(runs or P.RUNS)
    out = []
    for d in sorted((runs / 'history').glob('*')):
        if (d / 'summary.json').exists():
            try:
                pv = json.loads((d / 'provenance.json').read_text())
            except (OSError, ValueError):
                pv = {}
            out.append((d.name, d, pv))
    return out


def print_history(runs=None):
    hs = history_list(runs)
    if not hs:
        print('보관된 이전 결과가 없습니다. 채점(`bash frontend_benchmark/eval.sh score`)을 한 번 더 하면 그 전 결과가 보관됩니다.')
        return 2
    print(f'보관된 결과 {len(hs)}개 (<RUNS>/history):')
    for hid, d, pv in hs:
        print(f'  {hid}  {pv.get("archived_at", "–")}  시퀀스 {len(pv.get("sequences") or [])}개  '
              f'frontend src {str(pv.get("frontend_src_sha1") or "–")[:12]}  {pv.get("reason", "")}')
    print(f'\n비교: bash frontend_benchmark/eval.sh compare {hs[-1][0]}   (또는 latest)')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('which', nargs='?', default='latest', help='보관본 id 또는 latest (기본)')
    ap.add_argument('--list', action='store_true', help='보관 목록만 보기')
    ap.add_argument('--old', default=None, help='이전 summary.json 경로 (보관본 대신)')
    ap.add_argument('--new', default=None, help='지금 summary.json 경로 (기본 <RUNS>/summary.json)')
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if a.list:
        return print_history()
    new_p = Path(a.new) if a.new else P.RUNS / 'summary.json'
    if not new_p.exists():
        print(f'지금 채점 결과가 없습니다 ({new_p}) — 먼저 `bash frontend_benchmark/eval.sh score` 를 실행해 주세요.', file=sys.stderr)
        return 2
    old_info, new_info = {}, current_info()
    if a.old:
        old_p = Path(a.old)
        old_id = str(old_p.parent.name)
    else:
        hs = history_list()
        if not hs:
            print('비교할 이전 결과가 없습니다. 채점을 한 번 더 하면 그 전 결과가 `history/` 에 보관됩니다.', file=sys.stderr)
            print('지금까지의 결과는 `bash frontend_benchmark/eval.sh show` 로 볼 수 있습니다.', file=sys.stderr)
            return 2
        pick = hs[-1] if a.which in ('latest', '', None) else next((h for h in hs if h[0] == a.which or h[0].startswith(a.which)), None)
        if pick is None:
            print(f'그런 보관본이 없습니다: {a.which}', file=sys.stderr)
            print_history()
            return 2
        old_id, d, old_info = pick
        old_p = d / 'summary.json'
    old_rows, why = load_rows(old_p, '보관본 채점 결과')
    if old_rows is None:
        print(why, file=sys.stderr)
        return 2
    new_rows, why = load_rows(new_p, '지금 채점 결과')
    if new_rows is None:
        print(why, file=sys.stderr)
        return 2
    if not any(value(k, r) is not None for k in ROWS for r in list(old_rows.values()) + list(new_rows.values())):
        print(f'핵심 지표를 하나도 찾지 못했습니다 (보관본 {old_id}) — 옛 형식일 수 있습니다. '
              f'그 폴더의 summary.md 를 직접 보거나 지금 코드로 다시 채점해 주세요.', file=sys.stderr)
        return 2
    print(render_compare(old_rows, new_rows, old_id=old_id, old_info=old_info, new_info=new_info))
    return 0


if __name__ == '__main__':
    sys.exit(main())
