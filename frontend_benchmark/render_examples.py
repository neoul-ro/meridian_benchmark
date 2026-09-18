#!/usr/bin/env python3
"""3D 채점 판정을 눈으로 확인하는 그림. GT 트랙(ground-truth track) 하나 = 패널 하나.

글자와 색은 뷰어·summary.md 와 같은 곳에서 온다 (terms.py — 사용성 리뷰 m6). 한글은 Noto Sans CJK 로 그린다 (PIL).
패널: 그 GT 트랙이 crop 에 가장 크게 보인 keyframe 의 RGB(640×480 crop) 위에
  GT 물체 영역 윤곽 (seg 색 ∩ GT 트랙 점군 6cm 이내 ∩ 평가 거리 범위 5m) — 판정 색 (뷰어 맵과 같은 규칙, terms.COLORS):
    검출 파랑 · 미검출 주황 · 과소분할 청록 · 채점 제외 회색
  frontend 가 그 keyframe 에 publish(발행)한 예측 검출(predicted detection)의 점을 투영:
    파랑 = 이 GT 물체로 매칭 (뷰어의 TP 색) · 흰색 = 다른 GT 물체로 매칭 · 회색 = 매칭 없음(GT 에 없는 표면 등)
    윤곽 색(판정)과 점 색이 섞이지 않게, '다른 물체로 매칭' 은 판정 색이 아닌 흰색으로 그린다
  제목 = 판정 (검출 / 미검출 — 원인 표시 이름) · keyframe 프레임 · 가시 px · 그 keyframe 예측 검출 수
  그림 맨 아래 범례 한 줄 (legend_rows) — 색이 무슨 뜻인지 그림만 보고 알 수 있게
분류 (채점 대상(eligible) = max_px_crop ≥ score.json visibility.area_min_px 인 GT 트랙만)
  detected.jpg · missed_<원인 값>.jpg  — 원인 값은 score_episodes.csv 의 miss_reason 그대로 (표시 이름은 terms.py)
  summary.md 1장에서 이 폴더를 링크한다
매칭 열 = score.json params.tau_m 의 match@<τcm> (예전엔 match@20 고정)
오래된 그림 방지 (감사 C18): 입력(frontend_output.h5 · score_episodes.csv · score_observations.csv · score.json · gt_vis · GT h5 의 sha1)
  · 이 파일 코드 digest · 인자를 <out>/examples_provenance.json 에 기록한다. 같으면 다시 그리지 않고(--force 로 무시),
  다르면 그 폴더의 jpg 를 지우고 다시 그린다 (예전 원인 값 이름의 jpg 가 남아 섞이지 않게).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402  (meridian_benchmark 경로도 여기서 잡힘)
import provenance as PV  # noqa: E402
import terms as T  # noqa: E402
from meridian_benchmark.uhumans2 import UH2Sequence, unproject  # noqa: E402

CROP = 40
COLORS = T.COLORS
POINT_OTHER = (255, 255, 255)      # 다른 GT 물체로 매칭된 예측 점 — 판정 색과 헷갈리지 않게 흰색
FONTS = ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/truetype/nanum/NanumGothic.ttf')


def bgr(hex_color):
    """'#3987e5' → (229, 135, 57) BGR (cv2 용)."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def font(size=15):
    from PIL import ImageFont
    for f in FONTS:
        if Path(f).exists():
            try:
                return ImageFont.truetype(f, size)
            except OSError:
                pass
    return ImageFont.load_default()


def draw_text(img, items, size=15):
    """한글 글자를 그린다 (cv2.putText 는 한글을 못 그린다). items = [(x, y, 글자, BGR 색)]."""
    from PIL import Image, ImageDraw
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    fo = font(size)
    for x, y, text, col in items:
        d.text((x, y), text, font=fo, fill=(col[2], col[1], col[0]))
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def short_label(label):
    """제목 칸이 좁아서 영어 괄호는 뺀다 (범례에 온전한 이름이 있다)."""
    return label.split('(')[0].strip()


def verdict_of(ep_row, short=False):
    """→ (판정 표시 이름, terms.COLORS 키). 뷰어 맵과 같은 규칙: 미검출 원인의 분류(reason_group)로 색을 고른다."""
    if str(ep_row.get('detected')) == '1':
        return T.OBJECT['detected'], 'detected'
    code = ep_row.get('miss_reason') or ''
    grp = T.reason_group(code)
    name = short_label(T.reason_label(code)) if short else T.reason_label(code)
    label = f'{short_label(T.OBJECT["missed"])} — {name}' if code else T.OBJECT['missed']
    return label, ('merged' if grp == 'merged' else 'missed')


def panel_title(ep_row, frame, px, n_obs):
    """패널 제목 (한국어 표시 이름만 — 원시 라벨 MISS(merged) 같은 글자를 쓰지 않는다, m6)."""
    label, _ = verdict_of(ep_row, short=True)
    return (f'물체 {ep_row.get("gt_object_id", "?")} · 트랙 {ep_row.get("ep_index", "?")} · {label}'
            f' | kf 프레임 {frame} · 가시 {px}px · 예측 검출 {n_obs}개')


def legend_rows():
    """범례 [(글자, 색 hex)] — 글자는 terms.py, 색은 terms.COLORS (뷰어와 같은 색)."""
    return [(f'GT 영역 — {T.OBJECT["detected"]}', COLORS['detected']),
            (f'GT 영역 — {T.OBJECT["missed"]}', COLORS['missed']),
            (f'GT 영역 — {T.OBJECT["merged"]}', COLORS['merged']),
            (f'GT 영역 — {T.OBJECT["ineligible"]}', COLORS['ineligible']),
            ('예측 점 — 이 물체로 매칭', COLORS['detected']),
            ('예측 점 — 다른 물체로 매칭', '#ffffff'),
            ('예측 점 — 매칭 없음', COLORS['ineligible'])]


def legend_strip(width, height=54):
    """그림 맨 아래 범례 띠 (두 줄): GT 영역 색 · 예측 점 색."""
    img = np.full((height, width, 3), 20, np.uint8)
    rows = legend_rows()
    fo = font(14)
    width_of = (lambda s: int(fo.getlength(s))) if hasattr(fo, 'getlength') else (lambda s: int(len(s) * 11.5))
    items = []
    x, y = 8, 6
    for i, (label, col) in enumerate(rows):
        if i == 4:
            x, y = 8, 30
        cv2.rectangle(img, (x, y + 3), (x + 12, y + 15), bgr(col), -1)
        items.append((x + 18, y, label, (235, 235, 235)))
        x += 34 + width_of(label)
    return draw_text(img, items, size=14)


def panel(seq, gt, run, obs_by_kf, ep_row, vis, tau_key='match@20'):
    e = int(ep_row['ep_index']); tid = int(ep_row['gt_tracklet_id']); oid = int(ep_row['gt_object_id'])
    kf_frames = run['kf/frame_idx'][:]
    sel = (vis['ep_index'] == e) & np.isin(vis['frame'], kf_frames)
    if not sel.any():
        sel = vis['ep_index'] == e
    j = np.flatnonzero(sel)[np.argmax(vis['px_crop'][sel])]
    fi = int(vis['frame'][j])
    rgb = seq.rgb(fi)[:, CROP:CROP + 640].copy()
    gray = (cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) * 0.55).astype(np.uint8)   # 점 색이 보이게 배경은 어둡게 흑백
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    K = seq.camera_info()['K']
    t, q = seq.camera_poses(seq.stamps_ns[[fi]], cam='left_cam'); R = Rotation.from_quat(q[0]).as_matrix()
    # GT 영역
    ch = gt[f'tracklets/{tid:05d}/_metadata/color_hex'][()]; ch = ch.decode() if isinstance(ch, bytes) else ch
    seg = seq.seg_packed(fi); dep = seq.depth(fi)
    pts, (vv, uu) = unproject(dep, K, stride=1, valid_mask=(seg == int(ch, 16)))
    mask = np.zeros((480, 720), np.uint8)
    if len(pts):
        ok = np.linalg.norm(pts, axis=1) <= 5.0
        W = pts[ok] @ R.T + t[0]
        d = cKDTree(gt[f'tracklets/{tid:05d}/tracklet_geometry/points'][:]).query(W, distance_upper_bound=0.06)[0]
        mask[vv[ok][np.isfinite(d)], uu[ok][np.isfinite(d)]] = 255
    mask = mask[:, CROP:CROP + 640]
    # frontend 점
    k_idx = np.flatnonzero(kf_frames == fi)
    n_obs_here = 0
    if len(k_idx):
        k = int(k_idx[0])
        o0, n = int(run['kf/obs_start'][k]), int(run['kf/n_obs'][k])
        Kc = K.copy(); Kc[0, 2] -= CROP
        for o in range(o0, o0 + n):
            p = run['points'][run['obs/points_start'][o]:run['obs/points_start'][o] + run['obs/points_num'][o]]
            if not len(p):
                continue
            m = obs_by_kf.get(o)
            match = int(m[tau_key]) if m is not None and m[tau_key] not in ('', None) else 0
            col = bgr(COLORS['detected']) if match == oid else (POINT_OTHER if match > 0 else bgr(COLORS['ineligible']))
            c = (p - t[0]) @ R
            z = c[:, 2]; okz = z > 0.05
            u = (Kc[0, 0] * c[okz, 0] / z[okz] + Kc[0, 2]).astype(int); v = (Kc[1, 1] * c[okz, 1] / z[okz] + Kc[1, 2]).astype(int)
            inb = (u >= 0) & (u < 639) & (v >= 0) & (v < 479)
            for du in (0, 1):
                for dv in (0, 1):
                    img[v[inb] + dv, u[inb] + du] = col
            n_obs_here += 1
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _, ckey = verdict_of(ep_row)
    cv2.drawContours(img, cnts, -1, bgr(COLORS[ckey]), 2)
    title = panel_title(ep_row, fi, int(vis['px_crop'][j]), n_obs_here)
    cv2.rectangle(img, (0, 0), (640, 24), (0, 0, 0), -1)
    draw_text(img, [(5, 4, title, (245, 245, 245))], size=13)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--seq', required=True); ap.add_argument('--gt', required=True)
    ap.add_argument('--vis', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=4, help='분류마다 몇 개')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--force', action='store_true', help='입력이 같아도 다시 그림')
    a = ap.parse_args()
    run_dir = Path(a.run)
    need = [(run_dir / 'score.json', '채점 결과(score.json)', '먼저 채점해 주세요: bash frontend_benchmark/eval.sh score'),
            (run_dir / 'score_episodes.csv', '채점 결과(score_episodes.csv)', '먼저 채점해 주세요: bash frontend_benchmark/eval.sh score'),
            (run_dir / 'frontend_output.h5', 'frontend 출력', 'GPU=1 bash frontend_benchmark/eval.sh run <시퀀스>'),
            (Path(a.gt), 'GT h5', 'FB_GT 를 확인해 주세요'),
            (Path(a.vis), 'GT 가시성(gt_vis)', 'bash frontend_benchmark/eval.sh run <시퀀스> 이 만듭니다'),
            (Path(a.seq) / 'metadata.json', '데이터셋', 'FB_DATA 를 확인해 주세요')]
    miss = [f'{what} 없음 ({q}) — {todo}' for q, what, todo in need if not Path(q).exists()]
    if miss:                                                       # C4 · M10: 한 줄로 알리고 바로 멈춘다 (traceback 없이)
        print('[examples] 그림을 만들 수 없습니다: ' + '; '.join(miss[:2])
              + (f' (그 밖에 {len(miss) - 2}개 더 없음)' if len(miss) > 2 else ''), file=sys.stderr)
        return 2
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sj = json.loads((run_dir / 'score.json').read_text())
    area = int((sj.get('visibility') or {}).get('area_min_px') or 1600)
    tau_key = f"match@{int(round(float((sj.get('params') or {}).get('tau_m', 0.2)) * 100))}"
    H = PV.Hasher(P.RUNS / '.cache' / 'sha1.json')
    inputs = {k: H.file(v) for k, v in dict(frontend_h5=run_dir / 'frontend_output.h5', episodes=run_dir / 'score_episodes.csv',
                                           observations=run_dir / 'score_observations.csv', score=run_dir / 'score.json',
                                           vis=Path(a.vis), gt=Path(a.gt)).items()}
    prov = dict(inputs=inputs, code=PV.code_digest([Path(__file__)], [P.HERE, P.BENCH_PKG])[0], args=dict(n=a.n, seed=a.seed, seq=str(a.seq)))
    pp = out / 'examples_provenance.json'
    H.save()
    try:
        old = json.loads(pp.read_text())
    except (OSError, ValueError):
        old = None
    def complete():
        try:
            return all((out / f'{k}.jpg').exists() for k in json.loads((out / 'examples.json').read_text()))
        except (OSError, ValueError):
            return False
    if not a.force and old == prov and complete():
        print(f'[examples] {out} 최신 (입력·코드 같음) — 다시 그리려면 --force')
        return 0
    for j in out.glob('*.jpg'):
        j.unlink()
    seq = UH2Sequence(a.seq); gt = h5py.File(a.gt, 'r'); run = h5py.File(run_dir / 'frontend_output.h5', 'r')
    z = np.load(a.vis); vis = {k: z[k] for k in ('ep_index', 'frame', 'px_crop')}
    eps = [r for r in csv.DictReader(open(run_dir / 'score_episodes.csv')) if int(float(r.get('max_px_crop') or 0)) >= area]
    obs_by_kf = {int(r['obs']): r for r in csv.DictReader(open(run_dir / 'score_observations.csv'))}
    rng = np.random.default_rng(a.seed)
    groups = {'detected': [r for r in eps if r['detected'] == '1']}
    for code in sorted({r['miss_reason'] for r in eps if r['detected'] != '1'}, key=lambda c: (T.reason_order(c), c)):
        groups[f'missed_{code}'] = [r for r in eps if r['detected'] != '1' and r['miss_reason'] == code]
    manifest = {}
    for name, rows in groups.items():
        if not rows:
            continue
        pick = rng.choice(len(rows), size=min(a.n, len(rows)), replace=False)
        tiles = [panel(seq, gt, run, obs_by_kf, rows[i], vis, tau_key) for i in pick]
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
        grid = np.vstack([grid, legend_strip(grid.shape[1])])          # m6: 그림만 봐도 색 뜻을 알 수 있게
        p = out / f'{name}.jpg'
        cv2.imwrite(str(p), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
        manifest[name] = [int(rows[i]['ep_index']) for i in pick]
        print(f'[examples] {p} {manifest[name]} ' + ('' if name == 'detected' else T.reason_label(name[len('missed_'):])))
    (out / 'examples.json').write_text(json.dumps(manifest))
    pp.write_text(json.dumps(prov, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
