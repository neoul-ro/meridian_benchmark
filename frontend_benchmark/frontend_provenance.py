#!/usr/bin/env python3
"""frontend(GPU) 실행 결과의 출처 기록 · 확인 — eval.sh 가 frontend 를 다시 돌릴지 파일 시각(-nt)이 아니라 이것으로 정한다 (수정 ⑤ 후속 F1).

왜: run_frontend.py 의 docstring 만 고쳐도 파일 시각이 frontend_output.h5 보다 새로워져, 예전 eval.sh 는 GPU frontend(시퀀스당 1~4분)를
    다시 돌렸다 (09-17 감사 C 의 docstring 수정 뒤 실제로 그 상태였다).
기록  <run>/run_meta.json 의 provenance 키 (기존 키는 그대로)
  run_frontend_ast_sha1  run_frontend.py 의 docstring 을 뺀 AST sha1 (provenance.ast_digest — 주석·docstring 만 바뀌면 같다)
  frontend_src_sha1      frontend 소스 트리(src/meridian_frontend · src/meridian_frontend_msgs) 파일 내용 sha1 묶음.
                         (트리 이름/상대 경로, 파일 sha1) 을 경로순으로 모은 목록의 sha1. __pycache__ · *.pyc 는 뺀다
  frontend_src_files     그 파일 수
  engines                {파일 이름: {path, size, sha1}} — frontend 가 쓴 TensorRT 엔진
  dataset_seq            데이터셋 시퀀스 폴더 (절대 경로)
  recorded_at · backfill 기존 실행을 다시 돌리지 않고 기록만 채웠으면 {note, evidence}
확인  위 값을 지금 파일로 다시 계산해 기록과 비교한다. 다르면 이유(한국어 한 줄)를 보이고 종료 1, 같으면 '최신' · 종료 0.
      frontend_output.h5 가 없거나 기록이 없어도 종료 1.
큰 파일 sha1 은 provenance.Hasher 캐시((실제 경로, 크기, mtime_ns) → sha1)를 쓴다 (--cache, 기본 없음). --no-cache 면 매번 읽는다.
사용  python frontend_provenance.py check|record --run D --seq S --code run_frontend.py --src DIR [DIR ...] --engines F [F ...]
                                     [--cache sha1.json | --no-cache] [--backfill evidence.json]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance as PV  # noqa: E402

SKIP_DIRS = {'__pycache__'}
SKIP_SUFFIX = {'.pyc'}


def src_digest(roots, H):
    items = []
    for root in roots:
        root = Path(root)
        for f in sorted(p for p in root.rglob('*') if p.is_file()):
            rel = f.relative_to(root)
            if SKIP_DIRS & set(rel.parts) or f.suffix in SKIP_SUFFIX:
                continue
            items.append([f'{root.name}/{rel.as_posix()}', H.file(f)])
    return PV.key_of(items), len(items)


def current(code, src_roots, engines, seq, H):
    sha, n = src_digest(src_roots, H)
    return dict(
        run_frontend_ast_sha1=PV.sha1_text(PV.ast_dump(code)),
        frontend_src_sha1=sha, frontend_src_files=n,
        engines={Path(e).name: dict(path=str(Path(e).resolve()), size=Path(e).stat().st_size, sha1=H.file(e)) for e in engines},
        dataset_seq=str(Path(seq).resolve()))


def reasons(rec, cur):
    """기록과 지금 값의 차이 → 한국어 이유 목록 (빈 목록 = 최신)."""
    if not rec:
        return ['출처 기록 없음 (run_meta.json provenance)']
    out = []
    if rec.get('run_frontend_ast_sha1') != cur['run_frontend_ast_sha1']:
        out.append('run_frontend.py 코드(docstring 제외)가 바뀜')
    if rec.get('frontend_src_sha1') != cur['frontend_src_sha1']:
        out.append('frontend 소스(src/meridian_frontend*)가 바뀜')
    re_, ce = rec.get('engines') or {}, cur['engines']
    for name in sorted(set(re_) | set(ce)):
        a, b = re_.get(name) or {}, ce.get(name) or {}
        if a.get('sha1') != b.get('sha1') or a.get('path') != b.get('path'):
            out.append(f'엔진 {name} 이 바뀜')
    if rec.get('dataset_seq') != cur['dataset_seq']:
        out.append('데이터셋 시퀀스 경로가 바뀜')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cmd', choices=['check', 'record'])
    ap.add_argument('--run', required=True); ap.add_argument('--seq', required=True); ap.add_argument('--code', required=True)
    ap.add_argument('--src', nargs='+', required=True); ap.add_argument('--engines', nargs='+', required=True)
    ap.add_argument('--cache', default=None); ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--backfill', default=None, help='record: 다시 돌리지 않고 기록만 채울 때 근거 JSON ({note, evidence})')
    a = ap.parse_args(argv)
    run = Path(a.run)
    H = PV.Hasher(a.cache if a.cache and not a.no_cache else Path('/nonexistent/sha1.json'))
    if a.no_cache or not a.cache:
        H.d = {}
    missing = [p for p in [a.code, a.seq, *a.src, *a.engines] if not Path(p).exists()]
    if missing:
        print(f'입력 없음: {", ".join(missing)}')
        return 1
    cur = current(a.code, a.src, a.engines, a.seq, H)
    if a.cache and not a.no_cache:
        H.save()
    meta_p = run / 'run_meta.json'
    try:
        meta = json.loads(meta_p.read_text())
    except (OSError, ValueError):
        meta = {}
    if a.cmd == 'record':
        rec = dict(cur, recorded_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                   backfill=json.loads(Path(a.backfill).read_text()) if a.backfill else None)
        meta['provenance'] = rec
        run.mkdir(parents=True, exist_ok=True)
        PV.write_json_if_changed(meta_p, meta)
        print(f'출처 기록 {meta_p}' + (' (backfill)' if a.backfill else ''))
        return 0
    if not (run / 'frontend_output.h5').exists():
        print('frontend_output.h5 없음')
        return 1
    why = reasons(meta.get('provenance'), cur)
    print('최신' if not why else ' · '.join(why))
    return 1 if why else 0


if __name__ == '__main__':
    sys.exit(main())
