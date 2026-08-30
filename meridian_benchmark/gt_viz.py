"""Quick-look renders of the GT artifacts (PNG, no ROS needed).

Views:
  frame     rgb | gt_seg overlay side by side (one frame, or --range A B)
  tracklet  one large top-down PNG per GT tracklet from <gt>/gt_tracklets.h5:
            the whole GT map in grey (ceiling removed by a z filter), the
            tracklet's voxels painted with the camera RGB seen during its
            episode, boxed in red (--range A B over ids, --zmax, --px-cm).
            With --dataset the emptiest map-free corner also gets the rgb and
            class (seg_cam colour) images of the episode's largest observation,
            the object's mask outlined and boxed in red

Examples:
  python3 -m meridian_benchmark.gt_viz --gt <gtdir> --view frame --frame 400 \
      --dataset <sequence dir>
  python3 -m meridian_benchmark.gt_viz --gt <gtdir> --view tracklet
Output PNGs go to <gt>/viz/<view>/ unless --out is given.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

from .gt_tracklets import ceiling_cut, load_tracklets
import h5py
import matplotlib.pyplot as plt
import numpy as np

INK = '#333333'
MUTED = '#8a8a8a'


def _style(ax, title, sub):
    ax.set_title(title, color=INK, fontsize=12, loc='left', pad=22)
    ax.text(0, 1.008, sub, transform=ax.transAxes, color=MUTED, fontsize=9)
    ax.set_aspect('equal')
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color('#dddddd')
    ax.grid(color='#eeeeee', linewidth=0.6)
    ax.set_axisbelow(True)


def _oid_color(oid):
    """Stable identity colour: golden-ratio hue walk, so nearby ids differ."""
    h = (oid * 0.61803398875) % 1.0
    return plt.cm.hsv(h) * np.array([0.85, 0.85, 0.85, 1.0])


def _cam_path(gt):
    p = np.genfromtxt(gt / 'gt_poses.csv', delimiter=',', names=True)
    return np.stack([p['cam_tx'], p['cam_ty']], 1), p['stamp_ns'].astype(np.int64)


def view_frame(gt, dataset, i, out):
    import cv2
    rows = list(csv.DictReader(open(gt / 'frames.csv')))
    rgb = cv2.cvtColor(cv2.imread(str(Path(dataset) / rows[i]['rgb'])),
                       cv2.COLOR_BGR2RGB)
    seg = cv2.imread(str(gt / 'gt_seg' / f'{i:06d}.png'),
                     cv2.IMREAD_UNCHANGED)
    # map segment_id -> gt_object colour via segments.csv
    oid_of = {int(r['segment_id']): int(r['gt_object_id'])
              for r in csv.DictReader(open(gt / 'segments.csv'))
              if int(r['frame_idx']) == i}
    over = rgb.copy().astype(float)
    for sid, oid in oid_of.items():
        m = seg == sid
        over[m] = 0.35 * over[m] + 0.65 * np.array(_oid_color(oid)[:3]) * 255
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=110)
    for ax, img, t in ((axes[0], rgb, f'rgb — frame {i}'),
                       (axes[1], over.astype(np.uint8),
                        f'gt_seg overlay ({len(oid_of)} segments, '
                        'colour = gt_object)')):
        ax.imshow(img)
        ax.set_title(t, color=INK, fontsize=11, loc='left')
        ax.axis('off')
    f = out / f'frame_{i:06d}.png'
    fig.tight_layout()
    fig.savefig(f, facecolor='white')
    plt.close(fig)
    return f


def batch_frames(gt, dataset, a, b, out):
    """Fast rgb|overlay renders for frames [a, b) (cv2 only, no figure)."""
    import cv2
    from collections import defaultdict
    oid_of = defaultdict(dict)
    for r in csv.DictReader(open(gt / 'segments.csv')):
        oid_of[int(r['frame_idx'])][int(r['segment_id'])] = \
            int(r['gt_object_id'])
    rows = list(csv.DictReader(open(gt / 'frames.csv')))
    for i in range(a, b):
        rgb = cv2.imread(str(Path(dataset) / rows[i]['rgb']))
        seg = cv2.imread(str(gt / 'gt_seg' / f'{i:06d}.png'),
                         cv2.IMREAD_UNCHANGED)
        lut = np.zeros((256, 3), np.float32)
        for sid, oid in oid_of[i].items():
            lut[sid] = np.array(_oid_color(oid)[:3][::-1]) * 255  # BGR
        color = lut[seg]
        mask = (seg > 0)[..., None]
        over = np.where(mask, 0.35 * rgb + 0.65 * color, rgb).astype(np.uint8)
        tile = np.hstack([rgb, over])
        cv2.putText(tile, f'frame {i}  ({len(oid_of[i])} segments)',
                    (10, 24), 0, 0.7, (40, 40, 40), 2)
        cv2.imwrite(str(out / f'frame_{i:06d}.png'), tile)


def _topdown_base(gt, zmax, px_m, margin=1.0):
    """Grey top-down raster of the whole GT map below zmax (cached).

    Returns (image uint8 HxWx3, x0, y0, px_m). Higher voxels are painted last
    so furniture sits on top of the floor; shade darkens slightly with height
    so surfaces at different heights stay distinguishable while all grey."""
    clouds = np.load(gt / 'gt_object_clouds.npz')
    pts = np.concatenate([clouds[k] for k in clouds.files])
    x0, y0 = pts[:, 0].min() - margin, pts[:, 1].min() - margin
    W = int(np.ceil((pts[:, 0].max() + margin - x0) / px_m)) + 1
    H = int(np.ceil((pts[:, 1].max() + margin - y0) / px_m)) + 1
    cache = gt / 'viz' / f'_topdown_v2_{px_m * 100:g}cm_z{zmax:.2f}.npz'
    if cache.exists():
        c = np.load(cache)
        return c['img'], float(c['x0']), float(c['y0']), px_m
    keep = pts[:, 2] <= zmax
    p = pts[keep]
    order = np.argsort(p[:, 2])                       # low first, high last
    col = np.rint(np.interp(p[order, 2], [p[:, 2].min(), zmax], [245, 205]))
    u = ((p[order, 0] - x0) / px_m).astype(int)
    v = (H - 1 - (p[order, 1] - y0) / px_m).astype(int)   # y up
    img = np.full((H, W, 3), 255, np.uint8)
    c8 = col[:, None].astype(np.uint8)
    for du in (0, 1):            # 2x2 px per voxel: closes the sampling stripes
        for dv in (0, 1):
            img[np.clip(v + dv, 0, H - 1), np.clip(u + du, 0, W - 1)] = c8
    (gt / 'viz').mkdir(exist_ok=True)
    np.savez_compressed(cache, img=img, x0=x0, y0=y0)
    return img, x0, y0, px_m


def _paste_insets(img, occupancy, box, gt, dataset, frow, fi, sid, gap=12):
    """Representative rgb + class (seg_cam colour) images of frame `fi`, the
    object's mask outlined in red, pasted into the emptiest corner of the map
    that does not overlap the tracklet's red box."""
    import cv2
    rgb = cv2.imread(str(dataset / frow['rgb']))
    cls = cv2.imread(str(dataset / frow['seg']))
    mask = cv2.imread(str(gt / 'gt_seg' / f'{fi:06d}.png'), cv2.IMREAD_UNCHANGED) == sid
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    v, u = np.nonzero(mask)
    for im in (rgb, cls):
        # thin outline (masks seen through blinds/glass are many slivers) plus
        # a padded box so a small mask is still easy to find
        cv2.drawContours(im, cnts, -1, (0, 0, 255), 1)
        if len(u):
            cv2.rectangle(im, (max(u.min() - 6, 0), max(v.min() - 6, 0)),
                          (min(u.max() + 6, im.shape[1] - 1),
                           min(v.max() + 6, im.shape[0] - 1)), (0, 0, 255), 2)
    h, w = rgb.shape[:2]
    cap = 34
    panel = np.full((2 * (h + cap) + gap, w, 3), 255, np.uint8)
    for k, (im, label) in enumerate(((rgb, f'rgb frame {fi}'),
                                     (cls, f'class (seg_cam colour) frame {fi}'))):
        y = k * (h + cap + gap)
        cv2.putText(panel, label, (4, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (30, 30, 30), 2, cv2.LINE_AA)
        panel[y + cap:y + cap + h, :w] = im
    ph, pw = panel.shape[:2]
    H, W = occupancy.shape
    m = 20
    corners = [(m, m), (W - pw - m, m), (m, H - ph - m), (W - pw - m, H - ph - m)]
    def score(c):
        x, y = c
        if box is not None and not (x + pw < box[0] or x > box[2] or y + ph < box[1] or y > box[3]):
            return np.inf                      # would hide the tracklet
        return occupancy[y:y + ph, x:x + pw].sum()
    x, y = min(corners, key=score)
    img[y:y + ph, x:x + pw] = panel
    cv2.rectangle(img, (x - 2, y - 2), (x + pw + 1, y + ph + 1), (120, 120, 120), 2)


def batch_tracklets(gt, out, a=None, b=None, px_m=None, zmax=None,
                    dataset=None, log=print):
    """One large top-down PNG per tracklet: the whole GT map in grey (ceiling
    removed by the z filter), the tracklet's voxels painted with the camera
    RGB seen during its episode (_metadata/point_rgb), drawn regardless of
    z. A thin box marks the tracklet so small objects are easy to find."""
    import cv2
    clouds = np.load(gt / 'gt_object_clouds.npz')
    z = np.concatenate([clouds[k][:, 2] for k in clouds.files])
    if zmax is None:
        zmax = ceiling_cut(z)
    if px_m is None:
        span = max(np.ptp(np.concatenate([clouds[k][:, 0] for k in clouds.files])),
                   np.ptp(np.concatenate([clouds[k][:, 1] for k in clouds.files])))
        px_m = 0.02 if span / 0.02 > 3000 else 0.01
    base, x0, y0, px_m = _topdown_base(gt, zmax, px_m)
    H, W = base.shape[:2]
    r = max(2, int(round(0.04 / px_m)))   # tracklet voxel footprint in px (bold)
    # representative frame per (object, frame range): the observation with the
    # most pixels; needs the dataset for the rgb / seg_cam (class colour) images
    obs_by_oid = {}
    frame_rows = {}
    if dataset:
        for row in csv.DictReader(open(gt / 'segments.csv')):
            obs_by_oid.setdefault(int(row['gt_object_id']), []).append(
                (int(row['frame_idx']), int(row['n_pixels']), int(row['segment_id'])))
        frame_rows = {int(row['frame_idx']): row
                      for row in csv.DictReader(open(gt / 'frames.csv'))}
    occupancy = (base < 255).any(axis=2)
    with h5py.File(gt / 'gt_tracklets.h5', 'r') as f:
        ids = [int(n) for n in sorted(f['tracklets'])
               if (a is None or int(n) >= a) and (b is None or int(n) < b)]
    log(f'top-down {W}x{H} px @ {px_m * 100:g} cm/px, z<={zmax:.2f} m, {len(ids)} tracklets')
    for j, tid in enumerate(ids):
        meta, tr = load_tracklets(gt / 'gt_tracklets.h5', ids={tid})
        t = tr[tid]; md = t['_metadata']; oid = int(md['gt_object_id'])
        pts = t['tracklet_geometry']['points']
        rgb = np.asarray(md.get('point_rgb'), np.uint8) if md.get('point_rgb') is not None \
            and len(np.atleast_1d(md.get('point_rgb'))) else None
        img = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
        if len(pts):
            order = np.argsort(pts[:, 2])
            u = np.clip(((pts[order, 0] - x0) / px_m).astype(int), 0, W - 1)
            v = np.clip((H - 1 - (pts[order, 1] - y0) / px_m).astype(int), 0, H - 1)
            if rgb is not None and len(rgb) == len(pts):
                colr = rgb[order][:, ::-1]           # BGR
            else:
                c = np.array(_oid_color(oid)[:3]) * 255
                colr = np.tile(c[::-1].astype(np.uint8), (len(u), 1))
            for du in range(r):
                for dv in range(r):
                    img[np.clip(v + dv, 0, H - 1), np.clip(u + du, 0, W - 1)] = colr
            m = max(6, int(0.3 / px_m))
            x1, y1 = max(u.min() - m, 0), max(v.min() - m, 0)
            x2, y2 = min(u.max() + m, W - 1), min(v.max() + m, H - 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if dataset and oid in obs_by_oid:
            cand = [o for o in obs_by_oid[oid]
                    if md['first_frame'] <= o[0] <= md['last_frame']]
            if cand:
                fi, _, sid = max(cand, key=lambda o: o[1])
                box = (x1, y1, x2, y2) if len(pts) else None
                _paste_insets(img, occupancy, box, gt, Path(dataset),
                              frame_rows[fi], fi, sid)
        ext = (pts.max(0) - pts.min(0)) if len(pts) else np.zeros(3)
        lines = [
            f'tracklet {tid} -> object {oid} (color {md.get("color_hex", "")})  |  '
            f'frames {md["first_frame"]}-{md["last_frame"]} ({md["n_frames_observed"]} obs), '
            f'emitted {md["emit_frame"]}{" flushed" if md.get("flushed_at_end") else ""}',
            f'{len(pts):,} voxels @ 2cm, clip obs {md["n_obs_embedded"]}  |  '
            f'top-down, map z<={zmax:.2f} m, {px_m * 100:g} cm/px',
            f'points {len(pts):,} (raw observations {md["n_points_raw"]:,})  |  '
            f'size x {ext[0]:.2f} m (width), y {ext[1]:.2f} m (depth), z {ext[2]:.2f} m (height)',
            f'camera distance min {float(md.get("dist_min_m", np.nan)):.2f} m, mean {float(md.get("dist_mean_m", np.nan)):.2f} m'
            f'  |  {md.get("n_points_far", 0):,} raw points beyond {float(meta.get("max_range_m", np.nan)):g} m dropped',
        ]
        band = 60 * len(lines) + 20
        img = np.vstack([np.full((band, W, 3), 255, np.uint8), img])   # header band
        for k, line in enumerate(lines):
            cv2.putText(img, line, (16, 52 + 60 * k), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (30, 30, 30), 3, cv2.LINE_AA)
        cv2.imwrite(str(out / f'tracklet_{tid:05d}_obj{oid:05d}.png'), img,
                    [cv2.IMWRITE_PNG_COMPRESSION, 6])
        if (j + 1) % 200 == 0:
            log(f'  tracklets {j + 1}/{len(ids)}')
    log(f'{len(ids)} tracklet PNGs -> {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--gt', required=True)
    ap.add_argument('--view', required=True, choices=['frame', 'tracklet'])
    ap.add_argument('--frame', type=int, default=0)
    ap.add_argument('--range', type=int, nargs=2, metavar=('A', 'B'),
                    help='frame view: frames [A, B); tracklet view: ids [A, B)')
    ap.add_argument('--dataset', default='',
                    help='frame view: required; tracklet view: adds the rgb/class insets')
    ap.add_argument('--zmax', type=float, help='tracklet view: hide map above z (default p99)')
    ap.add_argument('--px-cm', type=float, help='tracklet view: cm per pixel (default 1 or 2)')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    gt = Path(a.gt).expanduser()
    out = (Path(a.out).expanduser() if a.out else gt / 'viz') / a.view
    out.mkdir(parents=True, exist_ok=True)
    if a.view == 'tracklet':
        batch_tracklets(gt, out, *(a.range or (None, None)),
                        px_m=(a.px_cm / 100 if a.px_cm else None), zmax=a.zmax,
                        dataset=a.dataset or None)
        return
    if not a.dataset:
        raise SystemExit('--view frame needs --dataset')
    if a.range:
        batch_frames(gt, a.dataset, a.range[0], a.range[1], out)
        print(f'frames [{a.range[0]}, {a.range[1]}) -> {out}')
    else:
        print(view_frame(gt, a.dataset, a.frame, out))


if __name__ == '__main__':
    main()
