#!/usr/bin/env python3
"""채점 → <RUNS>/summary.json · summary.md. 표의 숫자는 전부 각 채점기 결과 JSON·CSV 에서 온다.

  python report.py                      # paths.SEQUENCES 6개 전부. 없는 시퀀스는 summary.md 에 '누락' 으로 적고 종료 코드 2
  python report.py --seqs office_s1_06h apartment_00h   # 부분 실행 (요청한 것만 필요, 이름은 eval.sh 처럼 줄여 써도 됨)
  python report.py --force              # 캐시 무시

단계 그래프 (시퀀스마다). STEPS 순서 = 실행 순서, deps = 이 단계가 읽는 다른 단계의 산출물.
  labels      gt_labels_2d.build              GT 라벨 PNG (keyframe 마다)
  score3d     score_frontend.score            3D 검출 · 위치            deps: labels
              3D 매칭의 GT 표면 = 이 라벨 역투영 (labels_dir), keyframe 병렬 workers. 산출물에 score_kf_gt.csv
              캐시 키 params 에 라벨 출처(meta.json version · sha1)가 들어간다 — PNG 가 같아도 라벨을 다시 만들면 다시 채점
  mot         score_mot.score_sequence        keyframe 단위 MOT          deps: score3d (score_kf_gt.csv 의 present)
  difficulty  gt_difficulty_2d.build          Easy/Moderate/Hard        deps: labels
  seg2d       score_2d.score_sequence         2D 분할 PQ                deps: labels, difficulty
  geometry    score_geometry.score_sequence   같은 keyframe GT 표면 기하 deps: score3d, labels
  순서를 바꾸려면 deps 한 줄만 고치면 된다 (실행 순서는 deps 로 위상 정렬, 순환이면 멈춤).

캐시 (감사 C4 · C5 · C6 · C16) — mtime 이 아니라 내용으로 판단한다. 단계마다 키 =
  code    채점 모듈 + import 하는 로컬 모듈(이 폴더 · meridian_benchmark 의 uhumans2.py 등) 의 docstring 뺀 AST sha1
  inputs  GT h5 · frontend_output.h5 · gt_vis npz 의 sha1  (sha1 은 (경로, 크기, mtime_ns) 로 <RUNS>/.cache/sha1.json 에 캐시)
  paths   FB_GT · FB_DATA 에서 온 실제 경로 (다른 GT 폴더로 바꾸면 다시)
  params  이 보고서가 기대하는 채점 인자 = 채점 함수 시그니처 기본값
  deps    앞 단계 산출물의 sha1 묶음
  기록 위치  <run>/report_cache.json (단계별 키 · 산출물 sha1) · 라벨은 <gt_labels_2d>/<seq>/meta.json 의 report_provenance
  캐시로 쓰는 조건  키 같음 · 산출물이 전부 있고 sha1 이 기록과 같음 · 산출물 JSON 의 params 가 기대값과 같음
             (다른 인자로 CLI 를 직접 돌려 결과가 덮이면 다시 채점 — 감사 a08 R7)
  라벨      키가 다르면 폴더 전체를 다시 만든다 (PNG·meta.json 을 지우고 GL.build) — 옛·새 코드 결과가 섞이지 않게 (C5)
  변경 없음  아무 단계도 부르지 않고 meta.json · report_cache.json · sha1 캐시 · summary.* 를 다시 쓰지 않는다 (C16)
누락 시퀀스 (C9) 요청한 시퀀스에 frontend_output.h5 · gt_vis npz · GT h5 가 없으면 summary.md 에 이유와 함께 적고 종료 코드 2.
표시 (C15) 개수가 있으면 개수에서 한 번만 반올림 (half-up). JSON 에 비율만 있으면 그 값을 한 번 반올림.
"""
import argparse
import csv
import inspect
import json
import os
import re
import shutil
import sys
import time
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frontend_provenance as FP  # noqa: E402
import gt_difficulty_2d as GD  # noqa: E402
import gt_labels_2d as GL  # noqa: E402
import gt_surface as GS  # noqa: E402
import paths as P  # noqa: E402
import provenance as PV  # noqa: E402
import score_2d as S2  # noqa: E402
import score_frontend as S  # noqa: E402
import score_geometry as SG  # noqa: E402
import score_mot as SM  # noqa: E402
import status_md as STM  # noqa: E402  (판정 문구 · 한눈에 표를 status.md 와 한 곳에서)
import terms as T  # noqa: E402
import tide_rules as TR  # noqa: E402


def log(m):
    print(m, flush=True)


# ================================================================ 표시 (C15: 한 번만 반올림)
def _q(d, nd):
    return d.quantize(Decimal(1).scaleb(-nd), ROUND_HALF_UP)


def _dec(x):
    return x if isinstance(x, Decimal) else Decimal(repr(float(x)))


def num(x, nd=3):
    return '–' if x is None else str(_q(_dec(x), nd))


def pct(x, nd=1):
    return '–' if x is None else f'{_q(_dec(x) * 100, nd)}%'


def cm_of(m):
    """0.2 (m) → '20' (cm, 불필요한 0 없이)."""
    return format((_dec(m) * 100).normalize(), 'f')


def pct_kn(k, n, nd=1):
    return '–' if not n or k is None else f'{_q(Decimal(100 * int(k)) / Decimal(int(n)), nd)}%'


def recover(r, n):
    """4자리로 반올림된 비율 r 과 분모 n → 분자 k (n < 10000 이면 유일). 못 찾으면 None."""
    if r is None or not n:
        return None
    k = int(round(r * n))
    return k if abs(k / n - r) <= 5e-5 + 1e-12 else None


def pct_rn(r, n, nd=1):
    k = recover(r, n)
    return pct_kn(k, n, nd) if k is not None else pct(r, nd)


def g(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return default if d is None else d


# ================================================================ 난이도
def kitti_criteria():
    """gt_difficulty_2d.level_of 를 경계값으로 찔러서 등급 기준을 읽는다 (문서 숫자 = 코드 동작).
    → {easy|moderate|hard: {min_dim_gt: 폭이 이 값보다 커야, trunc_le: 잘림 상한, occ_le: 가림 비율 상한(없으면 제한 없음)}}"""
    out = {}
    for L, name in enumerate(('easy', 'moderate', 'hard')):
        c = {}
        m = next((m for m in range(0, 400) if GD.level_of(m, 0.0, 0.0) <= L), None)
        if m is not None:
            c['min_dim_gt'] = m - 1
        ts = [i / 100 for i in range(0, 101) if GD.level_of(1000, i / 100, 0.0) <= L]
        if ts:
            c['trunc_le'] = max(ts)
        os_ = [i / 100 for i in range(0, 101) if GD.level_of(1000, 0.0, i / 100) <= L]
        if os_ and max(os_) < 1.0:
            c['occ_le'] = max(os_)
        out[name] = c
    return out


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def area_min_of(run, default=1600):
    try:
        return int(json.loads((Path(run) / 'score.json').read_text())['visibility']['area_min_px'])
    except (OSError, ValueError, KeyError, TypeError):
        return default


def recall_3d_by_level(run, diff_npz, area_min_px=None):
    """난이도별 트랙 재현율 (KITTI 식 누적, 감사 C1).
    모집단 = 채점 대상(eligible) GT 트랙 = score_episodes.csv 의 max_px_crop ≥ area_min_px (score.json visibility.area_min_px).
    등급  = 그 GT 트랙의 keyframe 인스턴스 중 score_2d_gt.csv 에서 eligible(라벨 면적 ≥ 256칸) 인 행의 difficulty.npz 등급 최솟값.
            채점 대상 행이 없는 GT 트랙은 어느 등급에도 안 들어간다.
    → {easy|moderate|hard: {n, n_detected, recall(4자리, 호환용)}} — 표시는 n_detected/n 에서 한 번만 반올림 (C15)."""
    run = Path(run)
    area = area_min_px if area_min_px is not None else area_min_of(run)
    elig = {(int(r['ep_index']), int(r['frame'])) for r in read_csv(run / 'score_2d_gt.csv') if r.get('eligible') in ('True', '1', 'true')}
    z = np.load(diff_npz)
    best = {}
    for f, e, lv in zip(z['frame'], z['ep_index'], z['level']):
        if (int(e), int(f)) in elig:
            best[int(e)] = min(best.get(int(e), 9), int(lv))
    eps = [r for r in read_csv(run / 'score_episodes.csv') if int(float(r.get('max_px_crop') or 0)) >= area]
    out = {}
    for k, name in enumerate(('easy', 'moderate', 'hard')):
        sel = [r for r in eps if best.get(int(r['ep_index']), 9) <= k]
        nd = sum(str(r['detected']) == '1' for r in sel)
        out[name] = dict(n=len(sel), n_detected=nd, recall=round(nd / len(sel), 4) if sel else None)
    return out


def counts_3d(run, area):
    """score_episodes.csv 에서 트랙 재현율의 분자·분모 (C15 표시용). score.json 비율과 맞는지 consistent 로 표시."""
    eps = read_csv(Path(run) / 'score_episodes.csv')
    px = lambda r: int(float(r.get('max_px_crop') or 0))
    hum = lambda r: str(r.get('is_human', '0')) == '1'
    det = lambda rs: sum(str(r['detected']) == '1' for r in rs)
    c = lambda rs: dict(n=len(rs), detected=det(rs))
    el = [r for r in eps if px(r) >= area]
    return dict(all=c(eps), in_view=c([r for r in eps if px(r) > 0]), eligible=c(el),
                eligible_static=c([r for r in el if not hum(r)]), eligible_human=c([r for r in el if hum(r)]))


# ================================================================ 단계 그래프
def sig_defaults(fn):
    return {k: v.default for k, v in inspect.signature(fn).parameters.items() if v.default is not inspect.Parameter.empty}


def accepts(fn, name):
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _eq(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-9
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    return a == b


def params_mismatch(actual, expected):
    """expected 의 키 중 actual 에 있는 것만 비교 (출력에 없는 키는 판단 불가 → 비교 안 함)."""
    return [f'{k}={actual[k]!r}≠{v!r}' for k, v in expected.items() if k in actual and not _eq(actual[k], v)]


def expect_score3d(c=None):
    """이 보고서가 기대하는 score.json params (시그니처 기본값 · 코드 상수 · c 를 주면 라벨 출처).
    캐시 키의 params 로도 쓴다 → 라벨 meta.json 내용(version · sha1)이 키에 들어간다 (라벨 내용 출처)."""
    d = sig_defaults(S.score)
    e = {}
    for arg, key in (('tau', 'tau_m'), ('min_frac', 'min_frac'), ('min_frac', 'iou_threshold'), ('absorb_frac', 'absorb_frac'),
                     ('dup_policy', 'dup_policy')):
        if arg in d:
            e[key] = d[arg]
    if 'taus' in d and 'tau' in d:
        e['sensitivity_taus_m'] = sorted(set(list(d['taus']) + [d['tau']]))
    if hasattr(S, 'SIMILARITY'):
        e['similarity'] = S.SIMILARITY
    e['loc_min_iou'] = TR.LOC_MIN_IOU
    if c is not None:
        lm = GS.labels_meta(c.labels)
        e.update(labels_dir=lm['labels_dir'], labels_version=lm['version'], labels_meta_sha1=lm['meta_sha1'])
    return e


def check_score3d(ctx):
    try:
        j = json.loads((ctx.run / 'score.json').read_text())
    except (OSError, ValueError):
        return ['score.json 읽기 실패']
    bad = params_mismatch(j.get('params') or {}, expect_score3d(ctx))
    area = sig_defaults(S.score).get('area_min_px')
    if area is not None and g(j, 'visibility', 'area_min_px') not in (None, area):
        bad.append(f'visibility.area_min_px={g(j, "visibility", "area_min_px")}≠{area}')
    return bad


def check_mot(ctx):
    try:
        mp = json.loads((ctx.run / 'score_mot.json').read_text()).get('params') or {}
        sp = json.loads((ctx.run / 'score.json').read_text()).get('params') or {}
    except (OSError, ValueError):
        return ['score_mot.json/score.json 읽기 실패']
    keys = {'tau_m', 'min_frac', 'iou_threshold', 'labels_version', 'labels_meta_sha1'} | {k for k in mp if k.endswith('_column')}
    return [f'{k}={mp[k]!r}≠score.json {sp[k]!r}' for k in sorted(keys) if k in mp and k in sp and not _eq(mp[k], sp[k])]


def check_geometry(ctx):
    bad = check_json_params('score_geometry.json', SG.score_sequence, (('max_range', 'max_range_m'), ('stride', 'stride')))(ctx)
    try:
        gp = json.loads((ctx.run / 'score_geometry.json').read_text()).get('params') or {}
        tau = float((json.loads((ctx.run / 'score.json').read_text()).get('params') or {})['tau_m'])
    except (OSError, ValueError, KeyError, TypeError):
        return bad
    want = f'match@{int(round(tau * 100))}'
    if gp.get('match_column') not in (None, want):
        bad.append(f'match_column={gp.get("match_column")!r}≠{want!r}')
    return bad


def check_json_params(path, fn, mapping):
    def f(ctx):
        try:
            p = json.loads((ctx.run / path).read_text()).get('params') or {}
        except (OSError, ValueError):
            return [f'{path} 읽기 실패']
        d = sig_defaults(fn)
        return params_mismatch(p, {key: d[arg] for arg, key in mapping if arg in d})
    return f


def check_labels(ctx):
    try:
        m = json.loads((ctx.labels / 'meta.json').read_text())
    except (OSError, ValueError):
        return ['meta.json 없음']
    bad = []
    if hasattr(GL, 'LABEL_VERSION') and m.get('version') != GL.LABEL_VERSION:
        bad.append(f'version={m.get("version")!r}≠{GL.LABEL_VERSION!r}')
    d = sig_defaults(GL.build)
    return bad + params_mismatch(m, {k: d[k] for k in ('stride', 'max_range', 'tol', 'dominant', 'support_min') if k in d})


class Ctx:
    def __init__(self, s, workers):
        self.s = s; self.name = f'uHumans2_{s}'; self.workers = workers
        self.run = P.run_dir(s); self.gt = P.gt_h5(s); self.seqd = P.seq_dir(s); self.vis = P.vis_npz(s)
        self.h5 = self.run / 'frontend_output.h5'; self.labels = P.RUNS / 'gt_labels_2d' / self.name
        with h5py.File(self.h5, 'r') as f:
            self.kf_frames = f['kf/frame_idx'][:]
            self.has_masks = 'obs' in f and 'mask_bits' in f['obs']
        self.kf_digest = PV.sha1_text(','.join(str(int(x)) for x in self.kf_frames))

    def label_pngs(self):
        return [self.labels / f'{int(fi):06d}.png' for fi in sorted(set(int(x) for x in self.kf_frames))]


def run_labels(c):
    """라벨 폴더 전체를 다시 만든다 — meta.json 을 먼저 지워 도중에 죽어도 다음 실행이 다시 만들게."""
    c.labels.mkdir(parents=True, exist_ok=True)
    (c.labels / 'meta.json').unlink(missing_ok=True)
    for p in c.labels.glob('*.png'):
        p.unlink()
    GL.build(c.seqd, c.gt, c.kf_frames, c.labels, log=log)


def run_score3d(c):
    return S.score(c.run, c.seqd, c.gt, vis=c.vis, labels_dir=c.labels, workers=c.workers)


def STEPS(c):
    """단계 정의. inputs = 내용(sha1)으로 보는 파일 · paths = 경로 문자열로 보는 것 · deps = 앞 단계."""
    return [
        dict(name='labels', module=GL, deps=(), inputs=dict(gt_h5=c.gt), paths=dict(seq_dir=c.seqd, gt_h5=c.gt),
             extra=dict(kf_frames=c.kf_digest), params=lambda: {k: v for k, v in sig_defaults(GL.build).items() if k not in ('workers', 'log')},
             outputs=c.label_pngs, run=run_labels, check=check_labels, record='labels_meta'),
        dict(name='score3d', module=S, deps=('labels',), inputs=dict(frontend_h5=c.h5, gt_h5=c.gt, vis=c.vis),
             paths=dict(seq_dir=c.seqd, gt_h5=c.gt, vis=c.vis, labels=c.labels), params=lambda: expect_score3d(c),
             outputs=lambda: [c.run / 'score.json', c.run / 'score_episodes.csv', c.run / 'score_observations.csv', c.run / 'score_kf_gt.csv'],
             run=run_score3d, check=check_score3d),
        dict(name='mot', module=SM, deps=('score3d',), inputs=dict(frontend_h5=c.h5, gt_h5=c.gt, vis=c.vis), paths=dict(gt_h5=c.gt, vis=c.vis),
             params=lambda: {k: v for k, v in sig_defaults(SM.score_sequence).items() if k not in ('out_dir', 'score_dir')},
             outputs=lambda: [c.run / 'score_mot.json', c.run / 'score_mot_timeline.json'],
             run=lambda c: SM.score_sequence(c.run, c.gt, c.vis), check=check_mot),
        dict(name='difficulty', module=GD, deps=('labels',), inputs=dict(gt_h5=c.gt), paths=dict(seq_dir=c.seqd, gt_h5=c.gt),
             extra=dict(kf_frames=c.kf_digest), params=lambda: {k: v for k, v in sig_defaults(GD.build).items() if k not in ('workers', 'log')},
             outputs=lambda: [c.labels / 'difficulty.npz'], run=lambda c: GD.build(c.seqd, c.gt, c.labels, c.kf_frames, log=log),
             needs_masks=True),
        dict(name='seg2d', module=S2, deps=('labels', 'difficulty'), inputs=dict(frontend_h5=c.h5, gt_h5=c.gt), paths=dict(gt_h5=c.gt),
             params=lambda: {k: v for k, v in sig_defaults(S2.score_sequence).items() if k not in ('out_dir', 'log')},
             outputs=lambda: [c.run / 'score_2d.json', c.run / 'score_2d_gt.csv', c.run / 'score_2d_pred.csv'],
             run=lambda c: S2.score_sequence(c.run, c.gt, c.labels),
             check=check_json_params('score_2d.json', S2.score_sequence, (('area_min', 'area_min_cells'), ('theta', 'theta'))),
             needs_masks=True),
        dict(name='geometry', module=SG, deps=('score3d', 'labels'), inputs=dict(frontend_h5=c.h5), paths=dict(seq_dir=c.seqd),
             params=lambda: {k: v for k, v in sig_defaults(SG.score_sequence).items() if k not in ('out_dir', 'log')},
             outputs=lambda: [c.run / 'score_geometry.json', c.run / 'score_geometry.csv'],
             run=lambda c: SG.score_sequence(c.run, c.seqd, c.labels),
             check=check_geometry, needs_masks=True),
    ]


def topo(steps):
    by = {s['name']: s for s in steps}
    order, state = [], {}

    def visit(n, path):
        if state.get(n) == 2:
            return
        if state.get(n) == 1:
            raise SystemExit(f'단계 순환: {" → ".join(path + [n])}')
        state[n] = 1
        for d in by[n]['deps']:
            if d not in by:
                raise SystemExit(f'{n} 의 deps {d} 가 없는 단계')
            visit(d, path + [n])
        state[n] = 2
        order.append(by[n])
    for s in steps:
        visit(s['name'], [])
    return order


class Runner:
    def __init__(self, c, hasher, force):
        self.c, self.H, self.force = c, hasher, force
        self.cache_path = c.run / 'report_cache.json'
        try:
            self.records = json.loads(self.cache_path.read_text())
        except (OSError, ValueError):
            self.records = {}
        self.done = {}                                            # 이번 실행에서 확정된 단계 기록
        self.recomputed = []

    def _rel(self, p):
        p = Path(p)
        for base, tag in ((self.c.run, 'run'), (self.c.labels, 'labels')):
            try:
                return f'{tag}/{p.relative_to(base)}'
            except ValueError:
                pass
        return str(p)

    def parts(self, st):
        roots = [P.HERE, P.BENCH_PKG]
        code, _ = PV.code_digest([st['module'].__file__], roots)
        pr = dict(code=code)
        for role, p in st['inputs'].items():
            pr[f'input:{role}'] = self.H.file(p)
        pr['paths'] = PV.key_of({k: str(Path(v).resolve()) for k, v in st['paths'].items()})
        pr['params'] = PV.key_of(st['params']())
        for k, v in (st.get('extra') or {}).items():
            pr[f'extra:{k}'] = v
        for d in st['deps']:
            pr[f'dep:{d}'] = (self.done.get(d) or {}).get('out_digest')
        return pr

    def get_record(self, st):
        if st.get('record') == 'labels_meta':
            try:
                return json.loads((self.c.labels / 'meta.json').read_text()).get('report_provenance')
            except (OSError, ValueError):
                return None
        return self.records.get(st['name'])

    def put_record(self, st, rec):
        if st.get('record') == 'labels_meta':
            mp = self.c.labels / 'meta.json'
            m = json.loads(mp.read_text()) if mp.exists() else {}
            m['report_provenance'] = rec
            PV.write_json_if_changed(mp, m)
        else:
            self.records[st['name']] = rec
            PV.write_json_if_changed(self.cache_path, self.records)

    def why_stale(self, st, parts, rec):
        if self.force:
            return 'force'
        if not rec:
            return '기록 없음'
        diff = sorted(k for k in set(parts) | set(rec.get('parts', {})) if parts.get(k) != rec.get('parts', {}).get(k))
        if diff:
            return '바뀜: ' + ', '.join(diff)
        outs = [Path(p) for p in st['outputs']()]
        want = rec.get('outputs', {})
        if sorted(self._rel(p) for p in outs) != sorted(want):
            return '산출물 목록이 기록과 다름'
        missing = [self._rel(p) for p in outs if not p.exists()]
        if missing:
            return f'산출물 없음: {missing[:3]}'
        changed = [self._rel(p) for p in outs if self.H.file(p) != want[self._rel(p)]]
        if changed:
            return f'산출물 내용이 기록과 다름: {changed[:3]}'
        if st.get('check'):
            bad = st['check'](self.c)
            if bad:
                return 'params 불일치: ' + '; '.join(bad)
        return None

    def run(self, st):
        parts = self.parts(st)
        key = PV.key_of(parts)
        rec = self.get_record(st)
        why = self.why_stale(st, parts, rec)
        if why is None:
            self.done[st['name']] = rec
            return False
        log(f'[report] {T.seq_label(self.c.s)} {st["name"]} 다시 계산 — {why}')
        t0 = time.time()
        st['run'](self.c)
        outs = [Path(p) for p in st['outputs']()]
        missing = [str(p) for p in outs if not p.exists()]
        if missing:
            raise SystemExit(f'{T.seq_label(self.c.s)} {st["name"]}: 산출물이 안 생김 {missing[:3]}')
        out_sha = {self._rel(p): self.H.file(p) for p in outs}
        rec = dict(key=key, parts=parts, outputs=out_sha, out_digest=PV.key_of(sorted(out_sha.items())),
                   finished=time.strftime('%Y-%m-%d %H:%M:%S'), seconds=round(time.time() - t0, 1))
        if st.get('check'):
            bad = st['check'](self.c)
            if bad:
                log(f'[report] 경고 {T.seq_label(self.c.s)} {st["name"]}: 새로 채점했는데 params 가 기대값과 다름 — {bad}')
        self.put_record(st, rec)
        self.done[st['name']] = rec
        self.recomputed.append(st['name'])
        log(f'[report] {T.seq_label(self.c.s)} {st["name"]} 완료 ({rec["seconds"]}s)')
        return True


# ================================================================ 시퀀스 하나
def resolve_seq(q):
    q0 = q
    q = q[len('uHumans2_'):] if q.startswith('uHumans2_') else q
    if q in P.SEQUENCES:
        return q
    hit = [s for s in P.SEQUENCES if q.replace('_s1', '') in s.replace('_s1', '')]
    if len(hit) != 1:
        raise SystemExit(f'시퀀스 이름 모호/없음: {q0} ({" ".join(P.SEQUENCES)})')
    return hit[0]


def missing_reason(s):
    """채점 입력이 있는지 — 없으면 '무엇이 · 어디에 없고 · 다음에 무엇을 하면 되는지' 한 줄 (C4 · M10).
    데이터셋(uHumans2 언팩본)까지 본다: 없으면 라벨 단계의 자식 프로세스가 죽으면서 멈추기 때문이다."""
    ev = f'bash frontend_benchmark/eval.sh'
    need = [(P.seq_dir(s) / 'metadata.json', '데이터셋(uHumans2 언팩본)', f'FB_DATA 를 확인해 주세요 (지금 {P.DATA})'),
            (P.run_dir(s) / 'frontend_output.h5', 'frontend 출력', f'frontend 를 먼저 돌려 주세요: GPU=1 {ev} run {T.seq_label(s)}'),
            (P.gt_h5(s), 'GT h5', f'FB_GT 를 확인해 주세요 (지금 {P.GT_DIR})'),
            (P.vis_npz(s), 'GT 가시성(gt_vis)', f'{ev} run {T.seq_label(s)} 이 만듭니다')]
    miss = [f'{what} 없음 ({p}) — {todo}' for p, what, todo in need if not Path(p).exists()]
    return '; '.join(miss) or None


def _one_line(e):
    return str(e).splitlines()[0][:200] if str(e).strip() else type(e).__name__


def preflight(s):
    """라벨 단계 전에 입력을 부모 프로세스에서 한 번 열어 본다 (C4).
    multiprocessing Pool 의 initializer 가 자식에서 실패하면 Pool 이 워커를 끝없이 다시 띄워 화면이 멈추고 로그가 폭증한다
    (사용성 리뷰 e17: 148초 멈춤 · 로그 41MB). 같은 입력을 여기서 먼저 열어 한 줄로 알리고 끝낸다."""
    from meridian_benchmark.uhumans2 import UH2Sequence  # noqa: E402  (paths 가 경로를 잡은 뒤)
    seqd, gt, h5 = P.seq_dir(s), P.gt_h5(s), P.run_dir(s) / 'frontend_output.h5'
    try:
        UH2Sequence(seqd).camera_info()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f'{T.seq_label(s)}: 데이터셋을 읽을 수 없습니다 ({seqd}) — {_one_line(e)}. FB_DATA 를 확인해 주세요.')
    try:
        with h5py.File(gt, 'r') as f:
            f['index']['tracklet_id'][:1]
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f'{T.seq_label(s)}: GT h5 를 읽을 수 없습니다 ({gt}) — {_one_line(e)}. FB_GT 를 확인해 주세요.')
    try:
        with h5py.File(h5, 'r') as f:
            f['kf/frame_idx'][:1]
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f'{T.seq_label(s)}: frontend 출력을 읽을 수 없습니다 ({h5}) — {_one_line(e)}. '
                         f'GPU=1 bash frontend_benchmark/eval.sh run {T.seq_label(s)} 로 다시 만들어 주세요.')


def frontend_check(s):
    """C1 — 이 시퀀스의 frontend 출력이 지금 코드·frontend 소스·엔진·데이터셋과 맞는지 (frontend_provenance.py 와 같은 기준).
    → dict(status='ok'|'stale'|'unknown', reasons=[한국어 이유]). 확인할 입력이 없으면 unknown (채점을 막지 않는다)."""
    run = P.run_dir(s)
    engines = [P.MODELS / 'frontend_rtx3060' / n for n in ('fastsam.plan', 'clip_image.plan')]
    code = P.HERE / 'run_frontend.py'
    need = [code, P.FRONTEND_SRC, P.FRONTEND_MSGS, P.seq_dir(s), *engines]
    gone = [str(p) for p in need if not Path(p).exists()]
    if gone:
        return dict(status='unknown', reasons=[f'확인할 수 없습니다 — 없는 입력 {gone[0]}'])
    try:
        H = PV.Hasher(P.RUNS / '.cache' / 'sha1.json')
        cur = FP.current(code, [P.FRONTEND_SRC, P.FRONTEND_MSGS], engines, P.seq_dir(s), H)
        H.save()
        meta = json.loads((run / 'run_meta.json').read_text())
        why = FP.reasons(meta.get('provenance'), cur)
    except (OSError, ValueError, KeyError, TypeError) as e:
        return dict(status='unknown', reasons=[f'확인 실패 — {_one_line(e)}'])
    return dict(status='stale' if why else 'ok', reasons=why)


def merge_summary(old, new):
    """C2 — 부분 실행(--seqs)이 다른 시퀀스를 지우지 않게 합친다. 순서는 paths.SEQUENCES."""
    out = dict(old or {})
    out.update(new or {})
    order = {s: i for i, s in enumerate(P.SEQUENCES)}
    return {k: out[k] for k in sorted(out, key=lambda s: (order.get(s, 99), s))}


def benchmark_code_sha1():
    """채점 코드 전체(이 폴더의 채점기)의 docstring 뺀 AST digest — 보관 기록용."""
    files = [P.HERE / n for n in ('report.py', 'score_frontend.py', 'score_2d.py', 'score_geometry.py', 'score_mot.py',
                                  'gt_labels_2d.py', 'gt_difficulty_2d.py', 'match3d.py', 'tide_rules.py') if (P.HERE / n).exists()]
    try:
        return PV.code_digest(files, [P.HERE, P.BENCH_PKG])[0]
    except (OSError, SyntaxError, ValueError):
        return None


def frontend_src_sha1():
    try:
        H = PV.Hasher(P.RUNS / '.cache' / 'sha1.json')
        roots = [p for p in (P.FRONTEND_SRC, P.FRONTEND_MSGS) if Path(p).exists()]
        if not roots:
            return None
        sha = FP.src_digest(roots, H)[0]
        H.save()
        return sha
    except OSError:
        return None


def archive_history(reason='', runs=None):
    """C2 — 이전 summary.json · summary.md · status.md 를 <RUNS>/history/<시각>/ 에 출처와 함께 보관한다.
    보관할 이전 결과가 없으면 None. 내용이 그대로인 다시 실행에서는 main 이 부르지 않는다 (쌓이지 않게)."""
    runs = Path(runs or P.RUNS)
    if not (runs / 'summary.json').exists():
        return None
    stamp = time.strftime('%Y%m%d-%H%M%S')
    d = runs / 'history' / stamp
    n = 1
    while d.exists():
        d = runs / 'history' / f'{stamp}-{n}'
        n += 1
    d.mkdir(parents=True)
    for name in ('summary.json', 'summary.md', 'status.md'):
        if (runs / name).exists():
            shutil.copy2(runs / name, d / name)
    try:
        rows = json.loads((d / 'summary.json').read_text())
    except (OSError, ValueError):
        rows = {}
    scored = {s: max((st.get('finished') or '' for st in (r.get('report_steps') or {}).values()), default='')
              for s, r in rows.items()} if isinstance(rows, dict) else {}
    PV.write_json_if_changed(d / 'provenance.json', dict(
        archived_at=time.strftime('%Y-%m-%d %H:%M:%S'), reason=reason, sequences=sorted(rows) if isinstance(rows, dict) else [],
        scored_at=scored, frontend_src_sha1=frontend_src_sha1(), benchmark_code_sha1=benchmark_code_sha1(), runs=str(runs)))
    return d


def score_sequence(s, H, force, workers):
    preflight(s)                                                   # C4: 자식 프로세스에서 죽어 멈추기 전에 부모에서 한 번 연다
    c = Ctx(s, workers)
    R = Runner(c, H, force)
    for st in topo(STEPS(c)):
        if st.get('needs_masks') and not c.has_masks:
            log(f'[report] {T.seq_label(s)} {st["name"]} 건너뜀 — 마스크 없는 옛 실행 결과')
            continue
        R.run(st)
    sm = json.loads((c.run / 'score.json').read_text())
    area = int(g(sm, 'visibility', 'area_min_px', default=1600))
    runlog = P.LOGS / f'run_{s}.log'
    meta_p = c.run / 'run_meta.json'
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    sm.setdefault('frontend', {}).update(
        overflow_frames=len(re.findall(r'overflow: n_conf', runlog.read_text())) if runlog.exists() else None,
        frames=meta.get('n_frames'), ms_per_frame=round(meta['frontend_ms_mean'], 2) if meta.get('frontend_ms_mean') is not None else None)
    sm['counts_3d'] = counts_3d(c.run, area)
    if c.has_masks:
        sm['seg2d'] = json.loads((c.run / 'score_2d.json').read_text())
        sm['geometry'] = json.loads((c.run / 'score_geometry.json').read_text())
        sm['level_3d'] = recall_3d_by_level(c.run, c.labels / 'difficulty.npz', area)
        sm['level_3d_criteria'] = kitti_criteria()
    sm['mot'] = json.loads((c.run / 'score_mot.json').read_text())
    sm['report_steps'] = {n: dict(finished=r.get('finished'), key=r.get('key', '')[:16]) for n, r in R.done.items()}
    return sm, R.recomputed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--force', action='store_true', help='캐시 무시하고 전부 다시')
    ap.add_argument('--seqs', nargs='+', default=None, help='부분 실행할 시퀀스 (공백·쉼표 구분). 없으면 paths.SEQUENCES 전부')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--accept-stale', action='store_true',
                    help='frontend 출력이 지금 코드보다 오래돼도 채점하고 종료 코드 0 (표시는 남습니다). FB_ACCEPT_STALE=1 과 같음')
    ap.add_argument('--no-history', action='store_true', help='이전 결과를 history/ 에 보관하지 않음')
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    accept_stale = a.accept_stale or os.environ.get('FB_ACCEPT_STALE') == '1'
    t0 = time.time()
    try:
        requested = [resolve_seq(x) for q in a.seqs for x in q.split(',') if x] if a.seqs else list(P.SEQUENCES)
    except SystemExit as e:
        log(str(e))
        return 2
    H = PV.Hasher(P.RUNS / '.cache' / 'sha1.json')
    rows, missing, recomputed, stale = {}, {}, {}, {}
    try:
        for s in requested:
            why = missing_reason(s)
            if why:
                log(f'[report] {T.seq_label(s)} 누락 — {why}')
                missing[s] = why
                continue
            fc = frontend_check(s)
            if fc['status'] == 'stale':
                stale[s] = fc['reasons']
                log(f'[report] ⚠ 경고: {T.seq_label(s)} 의 frontend 출력이 지금 코드·소스와 다릅니다 — {", ".join(fc["reasons"])}')
            rows[s], recomputed[s] = score_sequence(s, H, a.force, a.workers)
            rows[s].setdefault('frontend', {})['provenance_check'] = fc   # 시각은 넣지 않는다 — 같은 입력이면 summary.json 이 그대로여야 한다 (C16)
    except KeyboardInterrupt:                                      # M8: 긴 traceback 대신 한 줄
        log('\n[report] 중단됨 (Ctrl+C) — 다시 실행하면 끝난 단계부터 이어서 합니다. summary 는 그대로 둡니다.')
        return 130
    except SystemExit as e:                                        # C4: preflight 등이 알린 한 줄
        if isinstance(e.code, str):
            log(f'[report] {e.code}')
            return 2
        raise
    finally:
        H.save()
    if not rows:
        log('[report] 채점할 시퀀스가 하나도 없습니다 — 위의 "누락" 줄이 무엇이 없는지 알려 줍니다.')
        log('[report] 이전 summary.md · summary.json 은 그대로 두었습니다 (지우지 않았습니다).')
        return 2
    old = {}
    sp = P.RUNS / 'summary.json'
    if sp.exists():
        try:
            old = json.loads(sp.read_text())
        except ValueError:
            old = {}
    merged = merge_summary(old, rows) if a.seqs else merge_summary({k: v for k, v in old.items() if k in rows}, rows)
    changed = json.dumps(merged, sort_keys=True, ensure_ascii=False) != json.dumps(old, sort_keys=True, ensure_ascii=False)
    if changed and not a.no_history and old:
        d = archive_history(f'채점 전 보관 (이번 실행: {", ".join(T.seq_label(s) for s in rows)})')
        if d:
            log(f'[report] 이전 결과 보관 → {d}  (이전과 비교: bash frontend_benchmark/eval.sh compare latest)')
    wrote = PV.write_json_if_changed(sp, merged)
    md = render(merged, missing, stale=stale, rescored=list(rows))
    wrote |= PV.write_text_if_changed(P.RUNS / 'summary.md', md)
    print(md)
    n_re = sum(len(v) for v in recomputed.values())
    log(f'[report] 다시 계산한 단계 {n_re}개 {({T.seq_label(k): v for k, v in recomputed.items() if v})} · summary {"갱신" if wrote else "변경 없음"} · '
        f'sha1 새로 읽음 {H.n_hashed}개 {H.bytes_hashed / 1e6:.1f}MB · {time.time() - t0:.1f}s')
    if missing:
        log(f'[report] 누락 시퀀스 {len(missing)}개: {", ".join(T.seq_label(s) for s in missing)} — 종료 코드 2 '
            f'(그 시퀀스만 빼고 채점했습니다. 부분 실행은 --seqs)')
        return 2
    if stale and not accept_stale:
        log(f'[report] ⚠ 오래된 frontend 출력 {len(stale)}개: {", ".join(T.seq_label(s) for s in stale)} — 종료 코드 4.')
        log('[report] 지금 코드로 다시 돌리려면 GPU=1 bash frontend_benchmark/eval.sh all · '
            '이대로 받아들이려면 --accept-stale (또는 FB_ACCEPT_STALE=1)')
        return 4
    return 0


# ================================================================ summary.md
def _params_lines(rows):
    """3D 매칭 설명 = score.json params (시퀀스마다 같으면 한 번만)."""
    keys = ('matching', 'tau_m', 'iou_threshold', 'min_frac', 'loc_min_iou', 'dup_policy', 'labels_version', 'presence', 'fp_rule',
            'dup_rule', 'merge_split_coverage', 'miss_reasons', 'detect_window')
    groups = {}
    for s, r in rows.items():
        p = r.get('params') or {}
        sig = PV.canon({k: p.get(k) for k in keys})
        groups.setdefault(sig, []).append(s)
    out = []
    for sig, seqs in groups.items():
        p = json.loads(sig)
        who = '' if len(groups) == 1 else f'({", ".join(seqs)}) '
        tau = f'{cm_of(p["tau_m"])}cm' if p.get('tau_m') is not None else '–'
        thr = p.get('iou_threshold', p.get('min_frac'))
        out.append(f'- {who}매칭 = `{p.get("matching")}` · dup_policy = {p.get("dup_policy")}')
        out.append(f'- {who}유사도 = 허용 거리 IoU_τ = P·R / (P + R − P·R), τ = {tau} (d < τ) · 매칭 후보 IoU_τ ≥ {thr} · '
                   f'GT 표면 = keyframe GT 라벨 역투영 (라벨 버전 `{p.get("labels_version")}`)')
        out.append(f'- {who}있음(present) = `{p.get("presence")}` · 미매칭 예측 FP / 제외 = `{p.get("fp_rule")}`')
        out.append(f'- {who}중복 = `{p.get("dup_rule")}` · Loc = 최고 IoU_τ {p.get("loc_min_iou")} 이상 {thr} 이하 (2D 와 같은 tide_rules.is_loc)')
        out.append(f'- {who}미검출 원인 = `{p.get("miss_reasons")}`')
        out.append(f'- {who}merged · split 의 덮음 = `{p.get("merge_split_coverage")}`')
    if len(groups) > 1:
        out.insert(0, '- **시퀀스마다 params 가 다르다** — 같은 표의 숫자를 서로 비교할 때 주의')
    return out


def sc(x):
    """0~1 점수 (PQ · F@τ · IDF1 · HOTA_α …) — 자리수는 terms.py 한 곳 (M5)."""
    return T.fmt_by_kind('score', x)


def cmv(x):
    """거리 — 표 칸은 단위 없이 (열 이름에 (cm) 이 있다)."""
    return T.fmt_by_kind('cm', x, unit=False)


def glance(rows, stale=None, rescored=None):
    """## 1. 한눈에 — status.md 와 같은 표 · 같은 판정 문구 (status_md.COLS · cells · verdict_bullets · JUDGE)."""
    stale = stale or {}
    L = ['## 1. 한눈에', '',
         '| ' + ' | '.join(STM.COLS) + ' | 채점 시각 | frontend 출력 |', '|---|' + '---:|' * (len(STM.COLS) - 1) + '---|---|']
    for s, r in rows.items():
        fin = max((st.get('finished') or '' for st in (r.get('report_steps') or {}).values()), default='–')
        pc = g(r, 'frontend', 'provenance_check', default={}) or {}
        mark = {'stale': '⚠ 오래됨', 'ok': '최신', 'unknown': '확인 안 함'}.get(pc.get('status'), '–')
        L.append(f'| {T.seq_label(s)} | ' + ' | '.join(STM.cells(r)) + f' | {fin[:16] or "–"} | {mark} |')
    L += [''] + STM.verdict_bullets(rows)
    L += [f'- 판정 기준 (우리가 정한 읽기용 기준, 가장 나쁜 시퀀스로 판정): 위치 {STM.JUDGE_RULE["position"]} · '
          f'분할 {STM.JUDGE_RULE["segmentation"]} · 추적 {STM.JUDGE_RULE["tracking"]}',
          f'- 지표 이름·자리수는 `terms.py` 한 곳에서 옵니다. 시퀀스 이름은 짧은 쪽으로 씁니다 (폴더는 `uHumans2_<시퀀스>`).']
    if rescored is not None and len(rescored) < len(rows):
        L.append(f'- 이번에 다시 채점한 시퀀스: {", ".join(T.seq_label(s) for s in rescored)} '
                 f'(나머지는 이전 채점 결과 그대로 — 위 "채점 시각" 열)')
    return L


def render(rows, missing=None, stale=None, rescored=None):
    missing = missing or {}
    stale = stale or {}
    first = rows[next(iter(rows))] if rows else {}
    fp = first.get('params') or {}
    area = g(first, 'visibility', 'area_min_px', default=1600)
    rng = fp.get('max_range_m')
    tau_cm = cm_of(fp['tau_m']) if fp.get('tau_m') is not None else '?'
    finished = max((st.get('finished') or '' for r in rows.values() for st in (r.get('report_steps') or {}).values()), default='')
    title = f'# frontend 평가 — uHumans2 {len(rows)}개 시퀀스' + (f' (누락 {len(missing)}개)' if missing else '')
    L = [title, '',
         f'생성: frontend_benchmark/report.py · 채점 결과 시각 {finished or "–"} (가장 최근에 끝난 채점 단계)', '',
         '조건: 카메라 pose = GT pose(ground-truth pose, SLAM 오차 없음) · 입력 드랍 0 · 엔진은 RTX 3060 에서 다시 빌드한 FP16', '']
    if stale:
        L += ['> ⚠ **오래된 frontend 출력으로 채점했습니다** — 아래 시퀀스는 지금 frontend 코드·소스와 다른 출력입니다. '
              '지금 코드로 보려면 `GPU=1 bash frontend_benchmark/eval.sh all` 을 실행해 주세요.', '']
        L += [f'> - {T.seq_label(s)}: {", ".join(why) or "출처 기록이 지금과 다름"}' for s, why in stale.items()] + ['']
    if missing:
        L += ['> **누락 시퀀스** — 아래 표에 없습니다:', ''] + [f'> - {T.seq_label(s)}: {why}' for s, why in missing.items()] + ['']
    if not rows:
        return '\n'.join(L) + '\n'
    L += glance(rows, stale, rescored)
    L += ['', '판정 그림 · 뷰어:', '',
          '- 물체 하나하나를 눈으로 보려면 뷰어 — `viewer/index.html` (`bash frontend_benchmark/eval.sh viewer` 가 만듭니다)',
          '- 판정 그림(검출 · 원인별 미검출 4장씩, 범례 포함): '
          + ' · '.join(f'[{T.seq_label(s)}](uHumans2_{s}/examples/)' for s in rows),
          '- 아래 2~7장은 같은 숫자를 자세히 본 것이고, 용어와 채점 기준 정의는 맨 끝 8장에 있습니다.']

    # 2. 미검출
    L += ['', '## 2. 미검출 — 트랙 재현율(track recall)', '',
          f'트랙 재현율 = GT 트랙 중, 있음(present: 그 keyframe 라벨 5m 안 {area}px 이상)인 keyframe 에서 IoU_τ ≥ '
          f'{fp.get("iou_threshold", fp.get("min_frac"))} 로 매칭된 예측 검출이 1개 이상인 비율. **굵은 열**이 채점 대상 기준.', '',
          f'| 시퀀스 | GT 트랙 | {T.metric_label("track_recall_all")} | {T.metric_label("track_recall_in_view")} | '
          '채점 대상 (정적+사람) | '
          f'**{T.metric_label("track_recall_eligible")}** | 정적 | 사람 | '
          f'미검출 트랙 원인 (정적 / 사람) | {T.metric_label("det_recall", population=False)} (정적 / 사람) |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---|---|']
    seen_reasons = set()
    for s, r in rows.items():
        m, v = r.get('missed') or {}, r.get('visibility') or {}
        bk, kk = v.get('by_kind') or {}, g(v, 'keyframe_level', 'by_kind', default={})
        c3 = r.get('counts_3d') or {}
        cell = lambda key, rate, n: (pct_kn(c3[key]['detected'], c3[key]['n']) if key in c3 and c3[key]['n'] == n else pct_rn(rate, n))
        st_, hu_ = bk.get('static') or {}, bk.get('human') or {}
        for d_ in (st_.get('miss_reasons_detectable'), hu_.get('miss_reasons_detectable')):
            seen_reasons |= set((d_ or {}).keys())
        L.append(f'| {T.seq_label(s)} | {m.get("n_gt_episodes", "–")} | {cell("all", m.get("recall"), m.get("n_gt_episodes"))} | '
                 f'{pct_rn(v.get("recall_in_view"), v.get("n_in_view"))} | '
                 f'{v.get("n_detectable", "–")} ({st_.get("n_detectable", "–")}+{hu_.get("n_detectable", "–")}) | '
                 f'**{cell("eligible", v.get("recall_detectable"), v.get("n_detectable"))}** | '
                 f'{cell("eligible_static", st_.get("recall_detectable"), st_.get("n_detectable"))} | '
                 f'{cell("eligible_human", hu_.get("recall_detectable"), hu_.get("n_detectable"))} | '
                 f'{T.reasons_text(st_.get("miss_reasons_detectable"))} / {T.reasons_text(hu_.get("miss_reasons_detectable"))} | '
                 f'{pct_rn(g(kk, "static", "recall"), g(kk, "static", "n"))} / {pct_rn(g(kk, "human", "recall"), g(kk, "human", "n"))} |')
    L += ['', '미검출 트랙 원인 (score_episodes.csv 의 miss_reason 값 → 표시 이름은 terms.py 한 곳에서):', '']
    for code in sorted(seen_reasons, key=lambda x: (T.reason_order(x), x)):
        gl = T.gloss(code)
        L.append(f'- {T.reason_label(code)} (`{code}`)' + (f' — 쉽게 말하면 {gl}. ' if gl else ': ') + T.reason_desc(code))
    kst = [(s, g(r, 'keyframe_gt', default={})) for s, r in rows.items() if g(r, 'keyframe_gt', 'status') is not None]
    if kst:
        cols = list(T.KF3D)
        L += ['', '### (keyframe, GT) 3D 상태 — score_kf_gt.csv (표면 있는 쌍 전부, 사람 포함)', '',
              f'present 쌍 = 그 keyframe 라벨 5m 안 {area}px 이상. 앞 다섯 열의 합 = present 쌍, 뒤 두 열은 present 아닌 쌍. '
              '미검출 원인의 판정 순서 과소분할 > 과다분할 > Loc > 미검출(Miss).', '',
              '| 시퀀스 | present 쌍 | ' + ' | '.join(T.kf3d_label(c) for c in cols) + ' |', '|---|---:|' + '---:|' * len(cols)]
        for s_, kg in kst:
            st_ = kg.get('status') or {}
            L.append(f'| {T.seq_label(s_)} | {kg.get("n_present", "–")} | ' + ' | '.join(str(st_.get(c, 0)) for c in cols) + ' |')
    eq = []
    for s, r in rows.items():
        kk = g(r, 'visibility', 'keyframe_level', 'by_kind', 'static', default={})
        mm = g(r, 'mot', 'metrics', default={})
        k = recover(kk.get('recall'), kk.get('n'))
        if 'TP' in mm and 'FN' in mm and kk.get('n'):
            eq.append((s, k == mm['TP'] and kk['n'] == mm['TP'] + mm['FN'], f'{k}/{kk["n"]}', f'{mm["TP"]}/{mm["TP"] + mm["FN"]}'))
    same = [e for e in eq if e[1]]
    diff = [e for e in eq if not e[1]]
    L += ['', f'검출 재현율(DetRe) = 있음(present: 그 keyframe 라벨 5m 안 {area}px 이상)인 (GT 트랙, keyframe) 쌍 중 그 keyframe 의 예측 검출과 매칭된 비율 '
              '(publish 타이밍과 분리해 본 검출 성능). '
          + (f'정적 물체 값은 5장 추적표의 TP/(TP+FN) 과 같습니다 ({len(same)}개 시퀀스 분자·분모 일치). ' if same else '')
          + ('추적표와 분자·분모가 다른 시퀀스: ' + ', '.join(f'{T.seq_label(s)} {a} vs 추적 {b}' for s, _, a, b in diff)
             + ' (추적의 이전 대응 유지 매칭 때문입니다). ' if diff else '')
          + 'score_mot.json 의 `DetRe` 키는 HOTA 매칭 기준 값이라 이것과 다를 수 있습니다.']
    if any('level_3d' in r for r in rows.values()):
        crit = next((r['level_3d_criteria'] for r in rows.values() if r.get('level_3d_criteria')), kitti_criteria())
        cells2d = next((g(r, 'seg2d', 'params', 'area_min_cells') for r in rows.values() if g(r, 'seg2d', 'params', 'area_min_cells')), 256)

        def crit_txt(c):
            t = [f'최소 bbox 변 > {c["min_dim_gt"]}px' if 'min_dim_gt' in c else None,
                 f'잘림 ≤ {pct(c["trunc_le"], 0)}' if 'trunc_le' in c else None,
                 f'가림 비율 ≤ {pct(c["occ_le"], 0)}' if 'occ_le' in c else None]
            return ' · '.join(x for x in t if x)
        L += ['', '### 난이도별 트랙 재현율 — Easy / Moderate / Hard (KITTI 기준 변형, 누적)', '',
              f'모집단 = 채점 대상 GT 트랙 (가시 {area}px 이상). 등급 = 그 GT 트랙의 keyframe 인스턴스 중 2D 채점 대상(라벨 면적 ≥ {cells2d}칸) 인 것의 '
              '가장 쉬운 등급 — 2D 채점 대상 인스턴스가 없으면 어느 등급에도 안 들어갑니다 (등급 없음).',
              f'Easy = {crit_txt(crit["easy"])} / Moderate = {crit_txt(crit["moderate"])} / Hard = {crit_txt(crit["hard"])} '
              '(최소 bbox 변 = min(w, h) · 가림 비율(occlusion ratio) · 기준 밖은 등급 없음, gt_difficulty_2d.level_of)', '',
              '| 시퀀스 | Easy | Moderate | Hard |', '|---|---:|---:|---:|']
        for s, r in rows.items():
            lv = r.get('level_3d')
            if lv:
                def lc(b):
                    if b.get('n_detected') is None:
                        return f'**{pct_rn(b.get("recall"), b.get("n"))}** (n={b.get("n")})'
                    return f'**{pct_kn(b["n_detected"], b["n"])}** ({b["n_detected"]}/{b["n"]})'
                L.append(f'| {T.seq_label(s)} | ' + ' | '.join(lc(lv[k]) for k in ('easy', 'moderate', 'hard')) + ' |')
    bins = g(first, 'visibility', 'recall_by_max_px_crop', default=[])
    if bins:
        lab = lambda b: '0 (가시 아님)' if str(b).startswith('0 ') else str(b)
        L += ['', '### 가시 픽셀 구간별 트랙 재현율 (GT 트랙 동안 crop 최대 가시 픽셀)', '',
              '| 시퀀스 | ' + ' | '.join(lab(b['bin']) for b in bins) + ' |', '|---|' + '---:|' * len(bins)]
        for s, r in rows.items():
            L.append(f'| {T.seq_label(s)} | ' + ' | '.join(f'{pct_rn(b.get("recall"), b.get("n"))} (n={b.get("n")})'
                                                          for b in g(r, 'visibility', 'recall_by_max_px_crop', default=[])) + ' |')

    # 3. 위치 · 기하
    ab = fp.get('absorb_frac')
    L += ['', '## 3. 매칭된 예측 검출의 위치 — GT 트랙 점군 기준', '',
          f'| 시퀀스 | 예측 검출 | {T.metric_label("obs_match_rate", population=False)} | '
          f'{T.metric_label("duplicate_rate", population=False)} | {T.metric_label("accuracy_track_cm", population=False)} p50 / p90 (cm) | '
          f'P@{tau_cm}cm | R@{tau_cm}cm (GT 트랙 p50) | 2cm 복셀 일치 | 다물체 섞임 |',
          '|---|---:|---:|---:|---|---:|---:|---:|---:|']
    for s, r in rows.items():
        a = r.get('accuracy') or {}
        n_obs = g(r, 'frontend', 'n_obs')
        L.append(f'| {T.seq_label(s)} | {n_obs if n_obs is not None else "–"} | {pct_rn(a.get("obs_match_rate"), n_obs)} | '
                 f'{pct_rn(a.get("duplicate_rate"), n_obs)} | '
                 f'{cmv(g(a, "dist_error_cm", "per_obs_mean", "p50"))} / {cmv(g(a, "dist_error_cm", "per_obs_mean", "p90"))} | '
                 f'{pct(a.get("within_tau_fraction_mean"))} | {pct(g(a, "episode_coverage_at_tau", "p50"))} | '
                 f'{pct(a.get("voxel_exact_2cm_mean"))} | {pct(a.get("multi_object_obs_rate"))} |')
    L += ['', '- 매칭률 = 매칭된 예측 검출 / 전체 예측 검출. 분모에 GT 가 정의하지 않은 표면(벽·바닥)과 중복 검출이 '
          '들어가므로 precision(정밀도)이 아닙니다',
          f'- 중복 검출(duplicate) = 매칭 안 된 예측 검출 중, 같은 keyframe 에서 다른 예측 검출이 이미 매칭한 present GT 와 IoU_τ ≥ '
          f'{fp.get("iou_threshold", fp.get("min_frac"))} 인 것 (TIDE Dupe, quantify.py:250-255). 추적에서는 오검출(FP)',
          f'- {T.metric_label("accuracy_track_cm")} = 예측 복셀 → **그 GT 트랙 점군**(GT h5 에 쌓인 그 물체 표면) 최근접 거리의 관측별 평균, '
          f'그 p50 / p90. 아래 기하 절의 accuracy 와 기준이 다릅니다',
          f'- P@{tau_cm}cm = 매칭된 예측 검출의 복셀 중 그 GT 트랙 점군 {tau_cm}cm 이내 비율의 평균 · '
          f'R@{tau_cm}cm = 탐지 인정된 예측 검출 복셀 합집합이 GT 트랙 점군을 {tau_cm}cm 안에 덮는 비율 (GT 트랙별, p50)',
          '- 2cm 복셀 일치 = 예측 복셀 중 GT 와 같은 2cm 칸에 있는 비율의 평균 · '
          f'다물체 섞임 = 매칭된 예측 검출 중, 가장 가까운 GT 표면이 2순위 GT 물체인 복셀이 {pct(ab, 0) if ab is not None else "absorb_frac"} 이상인 비율']
    pst = [(s, r.get('accuracy') or {}, g(r, 'frontend', 'n_obs')) for s, r in rows.items() if g(r, 'accuracy', 'unmatched_tide_errors') is not None]
    if pst:
        te = list(T.TIDE_ERR)
        L += ['', '### 예측 검출 상태 · 매칭 안 된 예측의 TIDE 오류', '',
              'TP · 무시 GT 매칭 · FP · void 겹침 무시 = score_observations.csv pred_status (FP / 무시 = panopticapi void 50% 규칙). '
              f'TIDE 오류는 매칭 안 된 예측마다 하나: 중복(Dupe) = 매칭된 present GT 와 IoU_τ ≥ {fp.get("iou_threshold", 0.5)} · '
              f'Loc = 최대 IoU_τ {fp.get("loc_min_iou", TR.LOC_MIN_IOU)} 이상 {fp.get("iou_threshold", 0.5)} 이하 · 배경(Bkg) = 그 미만 '
              '(TIDE quantify.py:228-265 순서, 무시 GT = present 아님).', '',
              '| 시퀀스 | 예측 검출 | TP | 무시 GT 매칭 | FP | void 겹침 무시 | ' + ' | '.join(T.tide_label(e) for e in te) + ' |',
              '|---|---:|---:|---:|---:|---:|' + '---:|' * len(te)]
        for s_, a, n_obs in pst:
            ut = a.get('unmatched_tide_errors') or {}
            L.append(f'| {T.seq_label(s_)} | {n_obs if n_obs is not None else "–"} | {a.get("n_obs_tp", "–")} | '
                     f'{a.get("n_obs_ignored_match", "–")} | '
                     f'{a.get("n_obs_fp", "–")} | {a.get("n_obs_excluded", "–")} | ' + ' | '.join(str(ut.get(e, 0)) for e in te) + ' |')
    geo = {k: r['geometry'] for k, r in rows.items() if r.get('geometry')}
    if geo:
        taus = next((g(x, 'params', 'taus_m') for x in geo.values() if g(x, 'params', 'taus_m')), [0.05, 0.10, 0.20])
        tk = [int(round(t * 100)) for t in taus]
        L += ['', '### 기하 — 같은 keyframe 에 실제로 보인 GT 표면 기준 (관측별 중앙값)', '',
              f'{T.metric_label("accuracy_keyframe_cm")} = 예측→GT 최근접 거리 평균 · completeness = GT→예측 최근접 거리 평균 · '
              'Chamfer-L1 = (accuracy + completeness) / 2 (비제곱) '
              '— 세 정의 모두 Occupancy Networks(Mescheder et al. CVPR 2019) eval.py. '
              '3장의 accuracy 와 이름은 같지만 기준(GT 트랙 점군 vs 같은 keyframe GT 표면)이 다르므로 값도 다릅니다.',
              'F@τ = 2·P@τ·R@τ / (P@τ + R@τ), P@τ = 예측 점 중 GT 까지 τ 미만 비율 · R@τ = GT 점 중 예측까지 τ 미만 비율 '
              '— Tanks and Temples(Knapitsch et al. 2017) evaluation.py 절차 (두 점군을 τ/2 복셀 다운샘플). '
              '완벽 예측 상한 = GT 라벨 픽셀을 frontend 격자 밀도로 넣었을 때의 값.', '',
              '| 시퀀스 | 채점 관측 | 건너뜀 | accuracy (cm) | completeness (cm) | Chamfer-L1 (cm) | '
              + ' | '.join(f'{"**" if t == tk[-1] else ""}F@{t}cm{"**" if t == tk[-1] else ""}' for t in tk) + f' | 완벽 예측 상한 F@{tk[-1]}cm |',
              '|---|---:|---:|---:|---:|---:|' + '---:|' * len(tk) + '---:|']
        for s, x in geo.items():
            q = x.get('summary') or {}
            med = lambda k: g(q, k, 'median')
            sk = x.get('skipped_obs') or {}
            alias = set((g(x, 'params', 'deprecated_skip_aliases') or {}).keys())
            skipped = ', '.join(f'{k} {v}' for k, v in sk.items() if v and k not in alias) or ('0' if 'n_skipped' in x else '–')
            L.append(f'| {T.seq_label(s)} | {x.get("n_obs", "–")} | {skipped} | {cmv(med("accuracy_cm"))} | {cmv(med("completeness_cm"))} | '
                     f'{cmv(med("chamfer_l1_cm"))} | '
                     + ' | '.join(f'{"**" if t == tk[-1] else ""}{sc(med(f"F@{t}"))}{"**" if t == tk[-1] else ""}' for t in tk)
                     + f' | {sc(g(x, "perfect_upper_bound", "summary", f"F@{tk[-1]}", "median"))} |')

    # 4. 분할
    seg = {k: r['seg2d'] for k, r in rows.items() if r.get('seg2d')}
    if seg:
        sp = next(iter(seg.values())).get('params') or {}
        th = sp.get('theta', 0.5)
        L += ['', '## 4. 분할 (2D, keyframe 이미지에서 frontend 마스크 vs GT 인스턴스)', '',
              f'- 매칭 = {sp.get("matching", "θ 마다 IoU>θ 쌍만 남긴 헝가리안 1:1")} · IoU = {sp.get("iou", "–")}',
              f'- 채점 대상 GT 인스턴스 = 라벨 면적 ≥ {sp.get("area_min_cells", "–")}칸 (칸 = 2.5×2.5px) · void(무시 라벨) = 라벨 0 · '
              f'void 겹침 무시 = {sp.get("ignore", "–")}',
              f'- PQ = SQ(매칭 쌍 평균 IoU) × RQ(TP / (TP + ½FP + ½FN)) — Kirillov et al. CVPR 2019 · panopticapi. 매칭 기준 IoU > {th}',
              '- 매칭 안 된 GT 의 원인 = 과소분할(under-segmentation, 여러 물체를 하나로 뭉쳐 봄) · '
              '과다분할(over-segmentation, 한 물체를 여러 조각으로 나눠 봄) · Loc(localization error) · 미검출(Miss). '
              '넷의 합이 FN 이고, 미검출(Miss) 열은 FN 전체가 아닙니다',
              f'- 과소분할 = 한 예측이 GT 2개 이상을 각각 50% 이상 덮음 · 과다분할 = {sp.get("split", "예측 2개 이상이 이 GT 안에 50% 이상")} · '
              f'Loc = 최고 IoU 가 {sp.get("loc_min_iou", TR.LOC_MIN_IOU)} 이상 {th} 이하 (3D 와 같은 tide_rules.is_loc, TIDE quantify.py:237)', '',
              f'| 시퀀스 | GT 인스턴스 | 재현율 @IoU0.25 / @IoU0.5 / @IoU0.75 | 정밀도 @IoU{th} | SQ | RQ | **PQ** | TP / FP / FN | '
              '과소분할 | 과다분할 | Loc | 미검출(Miss) |',
              '|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|']
        for k, g2 in seg.items():
            a2 = g2.get('all') or {}
            st = a2.get('gt_status') or {}
            rc = a2.get('recall') or {}
            n = a2.get('n_gt_instances')
            prec = pct_kn(a2['n_tp'], a2['n_tp'] + a2['n_fp']) if 'n_tp' in a2 and 'n_fp' in a2 and (a2['n_tp'] + a2['n_fp']) else pct(a2.get('precision'))
            tfn = f'{a2["n_tp"]} / {a2["n_fp"]} / {a2["n_fn"]}' if all(x in a2 for x in ('n_tp', 'n_fp', 'n_fn')) else '–'
            L.append(f'| {T.seq_label(k)} | {n if n is not None else "–"} | {pct_rn(rc.get("0.25"), n)} / {pct_rn(rc.get("0.5"), n)} / '
                     f'{pct_rn(rc.get("0.75"), n)} | '
                     f'{prec} | {sc(a2.get("SQ"))} | {sc(a2.get("RQ"))} | **{sc(a2.get("PQ"))}** | {tfn} | {st.get("merged", 0)} | '
                     f'{st.get("split", 0)} | {st.get("low_iou", 0)} | {st.get("missed", 0)} |')
        if any(g2.get('by_level') for g2 in seg.values()):
            L += ['', '### 난이도별 2D 재현율 @IoU0.5 (Easy / Moderate / Hard 누적, 위 3D 표와 같은 기준)', '',
                  '| 시퀀스 | Easy | Moderate | Hard |', '|---|---:|---:|---:|']
            for k, g2 in seg.items():
                b = g2.get('by_level')
                if b:
                    L.append(f'| {T.seq_label(k)} | ' + ' | '.join(f'**{pct_rn(g(b, n_, "recall", "0.5"), g(b, n_, "n"))}** (n={g(b, n_, "n")})'
                                                                   for n_ in ('easy', 'moderate', 'hard')) + ' |')

    # 5. 추적
    mot = {k: r['mot'] for k, r in rows.items() if r.get('mot')}
    if mot:
        mp = next(iter(mot.values())).get('params') or {}
        alpha = mp.get('min_frac', fp.get('min_frac'))
        L += ['', '## 5. 추적 — keyframe 단위 MOT (정적 물체)', '',
              '- timestep = publish 된 keyframe 하나. GT 트랙 = 정적 물체의 GT 트랙 — 시야를 벗어났다 돌아와 새 예측 ID 를 받는 것은 '
              '데이터 연관(data association)의 몫이라 세지 않습니다 (위키 Evaluation, 시퀀스 전체 물체 기준은 아래 참고 표)',
              f'- 있음(present) = {mp.get("present", f"그 keyframe 에 crop 가시 ≥ {area}px")} · 예측 검출 · 오검출(FP) = {mp.get("fp", "–")}',
              f'- 매칭 = {mp.get("match", "–")} — 이전 대응 유지(correspondence carry-over, MOT16 §4.1.1)',
              f'- HOTA_α · DetA_α · AssA_α = α = {alpha} 한 점의 값 ({mp.get("hota", "–")}). 논문의 최종 HOTA(α 0.05~0.95 평균)가 아닙니다',
              f'- MOTA = {T.metric_gloss("mota")}',
              f'- IDSW(ID switch) = {mp.get("idsw", "MOT16")} · Frag(fragmentation) = {mp.get("frag", "추적 끊김 횟수")} · '
              f'MT / PT / ML = GT 트랙이 있는 keyframe 중 매칭된 비율 ≥ 80% / 20% 이상 80% 미만 / 20% 미만 ({mp.get("mt", "MOT16 §4.1.6")})',
              f'- {T.metric_label("ids_per_track", population=False)} = 한 번 이상 매칭된 GT 트랙마다, 매칭된 서로 다른 예측 ID 개수의 평균', '',
              f'| 시퀀스 | GT 트랙 | **IDF1** | **HOTA_α** (DetA_α · AssA_α) | MOTA | TP / FN / FP | IDSW | Frag | MT / PT / ML | '
              f'{T.metric_label("ids_per_track", population=False)} |',
              '|---|---:|---:|---|---:|---|---:|---:|---|---:|']

        def idf1(m):
            if all(k in m for k in ('IDTP', 'IDFP', 'IDFN')) and (2 * m['IDTP'] + m['IDFP'] + m['IDFN']):
                return sc(Decimal(2 * m['IDTP']) / Decimal(2 * m['IDTP'] + m['IDFP'] + m['IDFN']))
            return sc(m.get('IDF1'))

        def mota(m):
            if all(k in m for k in ('FN', 'FP', 'IDSW', 'gt_detections')) and m['gt_detections']:
                return sc(1 - Decimal(m['FN'] + m['FP'] + m['IDSW']) / Decimal(m['gt_detections']))
            return sc(m.get('MOTA'))
        for k, g3 in mot.items():
            m = g3.get('metrics') or {}
            L.append(f'| {T.seq_label(k)} | {m.get("n_gt_tracks", "–")} | **{idf1(m)}** | **{sc(m.get("HOTA"))}** '
                     f'({sc(m.get("DetA"))} · {sc(m.get("AssA"))}) | '
                     f'{mota(m)} | {m.get("TP", "–")} / {m.get("FN", "–")} / {m.get("FP", "–")} | {m.get("IDSW", "–")} | {m.get("Frag", "–")} | '
                     f'{m.get("MT", "–")} / {m.get("PT", "–")} / {m.get("ML", "–")} | {T.fmt("ids_per_track", m.get("mean_ids_per_track"))} |')
        L += ['', '참고 — GT 트랙을 시퀀스 전체 물체 단위로 잡으면 (재방문해도 같은 예측 ID 요구 = 데이터 연관까지 포함한 기준):', '',
              f'| 시퀀스 | GT 트랙 (물체 단위) | IDF1 | HOTA_α | IDSW | {T.metric_label("ids_per_track", population=False)} |',
              '|---|---:|---:|---:|---:|---:|']
        for k, g3 in mot.items():
            m = g3.get('metrics_object') or {}
            L.append(f'| {T.seq_label(k)} | {m.get("n_gt_tracks", "–")} | {idf1(m)} | {sc(m.get("HOTA"))} | {m.get("IDSW", "–")} | '
                     f'{T.fmt("ids_per_track", m.get("mean_ids_per_track"))} |')

    # 6. 민감도 · 7. 실행 정보
    sens = g(first, 'sensitivity', default=[])
    L += ['', '## 6. τ 민감도 (트랙 재현율(전체 GT 트랙) / 매칭률)', '',
          '| 시퀀스 | ' + ' | '.join(f'τ={x.get("tau_cm")}cm' for x in sens) + ' |', '|---|' + '---|' * len(sens)]
    for s, r in rows.items():
        ne, no = g(r, 'missed', 'n_gt_episodes'), g(r, 'frontend', 'n_obs')
        L.append(f'| {T.seq_label(s)} | ' + ' | '.join(f'{pct_rn(x.get("recall"), ne)} / {pct_rn(x.get("obs_match_rate"), no)}'
                                                       for x in r.get('sensitivity') or []) + ' |')
    L += ['', '## 7. 실행 정보', '',
          '| 시퀀스 | 프레임 | keyframe (publish) | 예측 검출 | 예측 ID | frontend ms/프레임 | overflow 경고 프레임 | 범위 밖 제거 점 |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    dash = lambda x: '–' if x is None else x
    for s, r in rows.items():
        fr = r.get('frontend') or {}
        L.append(f'| {T.seq_label(s)} | {dash(fr.get("frames"))} | {dash(fr.get("n_kf"))} | {dash(fr.get("n_obs"))} | '
                 f'{dash(fr.get("n_tracklet_ids"))} | '
                 f'{dash(fr.get("ms_per_frame"))} | {dash(fr.get("overflow_frames"))} | {pct(fr.get("points_dropped_beyond_range"))} |')
    L += ['', 'overflow 경고 프레임 = frontend 로그의 `overflow: n_conf=… n_cand=…` 줄 수 (n_conf ≥ K1 또는 n_cand ≥ LANES 일 때 찍힘) · '
              '범위 밖 제거 점 = 평가 거리 범위 밖이라 뺀 예측 점의 비율']

    # 8. 용어 · params (정의는 맨 뒤 — M6)
    L += ['', '## 8. 용어와 채점 기준 (정의)', '',
          '자세한 정의와 JSON·CSV 키 대응은 `frontend_benchmark/docs/METRICS.md` 에 있습니다.', '',
          '- GT 트랙(ground-truth track) = 한 GT 물체가 연속으로 보인 구간 (GT h5 tracklet · CSV 의 ep_index)',
          '- 예측 검출(predicted detection) = frontend 가 keyframe 하나에 publish(발행)한 관측 하나 · 예측 ID(predicted ID) = frontend tracklet id',
          f'- 가시(visible) = 640×480 crop · 평가 거리 범위(evaluation range) {num(rng, 1) if rng is not None else "?"}m 안에서 그 GT 트랙 픽셀이 1개 이상',
          f'- 채점 대상(eligible) = 가시 픽셀이 GT 트랙 동안 한 프레임이라도 {area}px 이상 (frontend sam.AREA_MIN 256 proto 칸 × 2.5²). '
          '기준에 못 미치는 GT 는 채점 제외입니다',
          f'- {T.WORDS["ignored_match"]}',
          f'- {T.WORDS["void_overlap"]}',
          f'- {T.WORDS["out_of_range"]}', '',
          '### 3D 매칭 params — score.json params 에서 그대로 옮김 (정의와 출처는 docs/METRICS.md "3D 매칭 기준")', '']
    L += _params_lines(rows)
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    sys.exit(main())
