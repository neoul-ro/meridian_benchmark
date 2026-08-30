"""Build GT tracklets for the data-association benchmark.

Policy (confirmed 26/08/29): one tracklet = one continuous visibility episode
of one GT object. An object that is not observed for more than `gap` frames
(default 5; counted in frames — stamps are 50-200 ms apart) has left the view; its episode closes and a Tracklet is
emitted at the first frame where that is known (last observation + gap + 1).
Objects still visible at the end of the sequence are flushed at the last
frame. Tracklet ids increase by one in emission order.

  tracklet_geometry   only the surface the camera actually saw during the
                      episode: the GT frame clouds (strided unprojections of
                      the object's mask) accumulated over its frames and
                      voxelized on a map-frame-aligned 2 cm grid (cell centres)
  tracklet_semantics  the upstream CLIP node's pipeline (ViT-B/32,
                      mask_weighted_value pooling) run offline on each
                      observation (rgb + GT mask, same crop/preprocess/skip
                      rules as the live node), averaged over the episode and
                      re-normalized. Episodes whose observations were all
                      below the node's 16 px rule get an EMPTY float32[] (the
                      wire allows it; n_obs_embedded == 0 says why). Requires
                      the meridian_clip package and its models/; --no-embed
                      writes empty semantics everywhere.

File format: one HDF5 per sequence mirroring meridian_msgs/Tracklet; what is
not on the wire lives under `_metadata`.

  <gt>/gt_tracklets.h5
    _metadata/{gap_frames, voxel_m, embedding_model_id, embedding_backend,
               embedding_engine, embedding_engine_sha1, pooling_mode,
               n_tracklets, min_extent_m, max_range_m, ceiling_z_m,
               ceiling_margin_m, gt_config_hash, frame_id, built_at}
    index/{tracklet_id, gt_object_id, first_frame, last_frame, emit_frame,
           first_stamp_ns, last_stamp_ns, emit_stamp_ns, n_frames_observed,
           n_points, n_obs_embedded, flushed_at_end}   columnar [N], id order
           (fast "which tracklets are emitted at frame f" without a group scan)
    tracklets/<tid:05d>/tracklet_id                      uint32
    tracklets/<tid:05d>/tracklet_geometry/header/stamp_ns int64  last observation
    tracklets/<tid:05d>/tracklet_geometry/header/frame_id str    map
    tracklets/<tid:05d>/tracklet_geometry/points         float32 [M,3] voxel centres
    tracklets/<tid:05d>/tracklet_semantics               float32 [512]
    tracklets/<tid:05d>/_metadata/point_rgb            uint8 [M,3] mean camera RGB
        of every observation that fell in the voxel (rows align with points)
    tracklets/<tid:05d>/_metadata/{gt_object_id, color_hex, first_frame,
        last_frame, emit_frame, first_stamp_ns, last_stamp_ns, emit_stamp_ns,
        n_frames_observed, n_points_raw, n_points_far, n_obs_embedded,
        flushed_at_end, dist_min_m, dist_mean_m}
        (color_hex = the object's prefab colour from gt_objects.csv, handy for
        filtering structural prefabs such as floor/walls; observed points
        farther than max_range_m from the camera are dropped and counted in
        n_points_far; dist_* = camera->point distance over the episode: min
        of the per-frame minima, mean of the per-frame means)
  <gt>/gt_tracklets_index.csv   flat summary of the same fields

Load with `load_tracklets(path)` -> (file_metadata, {tid: nested dict}).
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np

HALF = 1 << 20   # voxel index range per axis (21 bits, signed)
DEFAULT_MODEL_DIR = str(Path(__file__).resolve().parents[2]
                        / 'meridian/meridian_clip/models')


# ------------------------------------------------------------- episodes ----

def episodes_from_segments(gt, gap):
    """[(oid, [frames...])] — visibility episodes split at gaps > `gap`."""
    frames = {}
    with open(gt / 'segments.csv') as f:
        for r in csv.DictReader(f):
            if int(r['n_pixels']) >= 1:
                frames.setdefault(int(r['gt_object_id']), set()).add(
                    int(r['frame_idx']))
    eps = []
    for oid in sorted(frames):
        fs = sorted(frames[oid])
        cur = [fs[0]]
        for a, b in zip(fs, fs[1:]):
            if b - a - 1 > gap:
                eps.append((oid, cur))
                cur = []
            cur.append(b)
        eps.append((oid, cur))
    return eps


# ------------------------------------------------------------ embedding ----

class ClipEmbedder:
    """Runs the upstream node's exact per-frame path offline."""

    def __init__(self, backend, model_dir, log):
        src = Path(__file__).resolve().parents[2] / 'meridian/meridian_clip'
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        import rclpy
        from meridian_clip.clip_inference_node import ClipInferenceNode
        rclpy.init(args=[])
        self._rclpy = rclpy
        self.node = ClipInferenceNode(backend=backend, model_dir=model_dir)
        n = self.node
        self.model_id = n.embedding_model_id
        self.dim = int(n.embedding_dim)
        self.pooling_mode = n.pooling_mode
        self.backend = backend
        # pin the exact weights: a rebuilt fp16 engine or the torch checkpoint
        # shifts vectors by cos ~2e-4, which must not pass as "the same"
        import hashlib
        eng = getattr(n, 'value_engine_path', None) or getattr(n, 'engine_path', None) \
            or getattr(n, 'model_path', None)
        self.engine = os.path.basename(str(eng)) if eng else 'unknown'
        try:
            self.engine_sha1 = hashlib.sha1(open(str(eng), 'rb').read()).hexdigest()[:12]
        except Exception:
            self.engine_sha1 = 'unknown'
        log(f'  clip: {self.model_id} dim={self.dim} backend={backend} '
            f'pooling={n.pooling_mode} crop={n.crop_policy}/{n.crop_fit}')

    def embed(self, rgb, labels):
        """-> {segment_id: embedding[512]} for the segments the node encodes."""
        n = self.node
        sids, regions, masks, _ = n.build_regions(
            rgb_image=rgb, labels=labels, with_masks=n.needs_region_masks)
        if not sids:
            return {}
        prepared = n.backend.prepare(regions=regions, masks=masks,
                                     pooling_mode=n.pooling_mode)
        emb = n.backend.run(prepared=prepared, batch_size=n.batch_size,
                            normalize=n.normalize_embeddings,
                            gamma=n.patch_weight_gamma,
                            min_patch_occupancy=n.min_patch_occupancy,
                            empty_mask_fallback=n.empty_mask_fallback)
        emb = n.apply_alignment(emb)
        stats = n.backend.last_pooling_stats
        if stats is not None and not bool(stats.keep.all()):
            sids = [s for s, k in zip(sids, stats.keep) if k]
        return {int(s): np.asarray(e, np.float32) for s, e in zip(sids, emb)}

    def close(self):
        self.node.destroy_node()
        self._rclpy.shutdown()


# ---------------------------------------------------------------- build ----

def _pack(idx):
    if len(idx) and int(np.abs(idx).max()) >= HALF:
        raise ValueError('voxel index out of the 21-bit packing range; '
                         'coordinates or voxel size are off')
    return ((idx[:, 0] + HALF) << 42) | ((idx[:, 1] + HALF) << 21) | (idx[:, 2] + HALF)


def _unpack(keys, voxel):
    ix = (keys >> 42) & 0x1FFFFF
    iy = (keys >> 21) & 0x1FFFFF
    iz = keys & 0x1FFFFF
    idx = np.stack([ix, iy, iz], axis=1).astype(np.float64) - HALF
    return ((idx + 0.5) * voxel).astype(np.float32)


def _camera(gt, dataset_dir):
    """K and per-frame world_T_camera (R[N,3,3], t[N,3]) for reprojection."""
    import yaml
    from scipy.spatial.transform import Rotation
    with open(Path(dataset_dir) / 'camera_info/left_cam.yaml') as f:
        K = np.array(yaml.safe_load(f)['K'], np.float64).reshape(3, 3)
    pose = np.genfromtxt(gt / 'gt_poses.csv', delimiter=',', names=True)
    t = np.stack([pose[f'cam_t{a}'] for a in 'xyz'], 1)
    q = np.stack([pose[f'cam_q{a}'] for a in 'xyzw'], 1)
    return K, Rotation.from_quat(q).as_matrix(), t


def _pixel_rgb(pts, rgb, K, R, t):
    """World points -> RGB sampled at their (rounded) pixel; the points were
    unprojected from integer pixels with the same pose, so this is exact."""
    pc = (pts.astype(np.float64) - t) @ R          # R^T (p - t)
    z = np.maximum(pc[:, 2], 1e-6)
    u = np.clip(np.rint(pc[:, 0] / z * K[0, 0] + K[0, 2]).astype(int), 0, rgb.shape[1] - 1)
    v = np.clip(np.rint(pc[:, 1] / z * K[1, 1] + K[1, 2]).astype(int), 0, rgb.shape[0] - 1)
    return rgb[v, u]


def build(gt_dir, dataset_dir, gap=5, voxel=0.02, embed=True,
          backend='tensorrt', model_dir=DEFAULT_MODEL_DIR, max_frames=0,
          world_frame='map', recolor_only=False, min_extent=0.04, max_range=5.0,
          ceiling_margin=0.3, log=print):
    gt = Path(gt_dir).expanduser()
    with open(gt / 'frames.csv') as f:
        frames = list(csv.DictReader(f))
    stamps = np.array([int(r['stamp_ns']) for r in frames], np.int64)
    n_frames = len(frames) if not max_frames else min(len(frames), max_frames)
    K, R_wc, t_wc = _camera(gt, dataset_dir)
    if recolor_only:
        embed = False

    eps = [(oid, [f for f in fs if f < n_frames])
           for oid, fs in episodes_from_segments(gt, gap)]
    eps = [e for e in eps if e[1]]
    # emission frame + id order
    emitted = []
    for oid, fs in eps:
        emit = fs[-1] + gap + 1
        flushed = emit > n_frames - 1
        emitted.append({'oid': oid, 'frames': fs, 'first': fs[0],
                        'last': fs[-1], 'emit': min(emit, n_frames - 1),
                        'flushed': flushed})
    emitted.sort(key=lambda e: (e['emit'], e['oid']))
    for k, e in enumerate(emitted):
        e['tid'] = k + 1
        e['vox'] = []
        e['rgb'] = []      # (keys, rgb uint8) per frame, averaged per voxel at the end
        e['n_raw'] = 0
        e['n_far'] = 0      # raw points beyond max_range (dropped)
        e['dist'] = []      # per observed frame: (min, mean) camera->point distance
        e['emb_sum'] = None
        e['n_emb'] = 0
    active = {}   # frame -> {oid: episode}
    for e in emitted:
        for f in e['frames']:
            active.setdefault(f, {})[e['oid']] = e
    log(f'[{gt.name}] {len(emitted)} tracklets from {len(set(e["oid"] for e in emitted))} '
        f'objects (gap={gap}, {sum(e["flushed"] for e in emitted)} flushed at end)')

    clip = None
    if embed:
        clip = ClipEmbedder(backend, model_dir, log)
        dim = clip.dim
    else:
        dim = 512

    t0 = time.time()
    for i in range(n_frames):
        act = active.get(i)
        if not act:
            continue
        d = np.load(gt / 'gt_frame_clouds' / f'{i:06d}.npz')
        off, all_pts = d['offsets'], d['points']   # decompress once per frame
        rgb = cv2.cvtColor(cv2.imread(str(Path(dataset_dir) / frames[i]['rgb'])),
                           cv2.COLOR_BGR2RGB)
        sid_to_oid = {}
        for k, (sid, oid) in enumerate(zip(d['segment_ids'].tolist(),
                                            d['gt_object_ids'].tolist())):
            sid_to_oid[int(sid)] = int(oid)
            e = act.get(int(oid))
            if e is None:
                continue
            pts = all_pts[off[k]:off[k + 1]]
            if len(pts):
                dist = np.linalg.norm(pts - t_wc[i], axis=1)
                near = dist <= max_range
                e['n_far'] += int((~near).sum())
                pts, dist = pts[near], dist[near]
            if len(pts):
                e['n_raw'] += len(pts)
                e['dist'].append((float(dist.min()), float(dist.mean())))
                keys = _pack(np.floor(pts.astype(np.float64) / voxel).astype(np.int64))
                e['vox'].append(np.unique(keys))
                e['rgb'].append((keys, _pixel_rgb(pts, rgb, K, R_wc[i], t_wc[i])))
                if len(e['vox']) >= 32:
                    e['vox'] = [np.unique(np.concatenate(e['vox']))]
        if clip is not None:
            labels = cv2.imread(str(gt / 'gt_seg' / f'{i:06d}.png'),
                                cv2.IMREAD_UNCHANGED)
            for sid, vec in clip.embed(rgb, labels).items():
                e = act.get(sid_to_oid.get(sid))
                if e is None:
                    continue
                e['emb_sum'] = vec if e['emb_sum'] is None else e['emb_sum'] + vec
                e['n_emb'] += 1
        if (i + 1) % 500 == 0:
            log(f'  frame {i + 1}/{n_frames} ({time.time() - t0:.0f}s)')
    if clip is not None:
        clip.close()

    def voxel_rgb(e, keys_sorted):
        """Mean RGB per voxel, rows aligned with the sorted unique keys."""
        if not e['rgb']:
            return np.zeros((len(keys_sorted), 3), np.uint8)
        k = np.concatenate([a for a, _ in e['rgb']])
        c = np.concatenate([b for _, b in e['rgb']]).astype(np.float64)
        pos = np.searchsorted(keys_sorted, k)
        acc = np.zeros((len(keys_sorted), 3)); cnt = np.zeros(len(keys_sorted))
        np.add.at(acc, pos, c); np.add.at(cnt, pos, 1)
        return np.rint(acc / np.maximum(cnt, 1)[:, None]).astype(np.uint8)

    if recolor_only:
        out = gt / 'gt_tracklets.h5'
        n_ok = 0
        with h5py.File(out, 'r+') as f:
            for e in emitted:
                g = f[f'tracklets/{e["tid"]:05d}']
                keys = np.unique(np.concatenate(e['vox'])) if e['vox'] else np.zeros(0, np.int64)
                if g['tracklet_geometry/points'].shape[0] != len(keys):
                    raise RuntimeError(f'tracklet {e["tid"]}: point count differs from '
                                       'the stored file; rebuild instead of --recolor')
                if 'point_rgb' in g['_metadata']:
                    del g['_metadata/point_rgb']
                g.create_dataset('_metadata/point_rgb', data=voxel_rgb(e, keys),
                                 compression='gzip', compression_opts=4)
                n_ok += 1
        log(f'[{gt.name}] recolored {n_ok} tracklets in {out} ({time.time() - t0:.0f}s)')
        return out

    # drop degenerate tracklets (voxel extent <= min_extent on any axis,
    # i.e. a single voxel sheet/line) and renumber ids in emission order
    ceil_z = ceiling_z(gt)
    emitted = _drop_thin(emitted, voxel, min_extent, ceil_z, ceiling_margin,
                         log, gt.name)

    # ---------------- write ----------------
    src_cfg = json.load(open(gt / 'config.json')) if (gt / 'config.json').exists() else {}
    color_of = {}
    if (gt / 'gt_objects.csv').exists():
        with open(gt / 'gt_objects.csv') as f:
            color_of = {int(r['gt_object_id']): r['color_hex'] for r in csv.DictReader(f)}
    out = gt / 'gt_tracklets.h5'
    sdt = h5py.string_dtype()
    rows = []
    with h5py.File(out, 'w') as f:
        meta = {'gap_frames': np.int32(gap), 'voxel_m': float(voxel),
                'embedding_model_id': clip.model_id if clip else 'none',
                'embedding_backend': clip.backend if clip else 'none',
                'embedding_engine': clip.engine if clip else 'none',
                'embedding_engine_sha1': clip.engine_sha1 if clip else 'none',
                'pooling_mode': clip.pooling_mode if clip else 'none',
                'n_tracklets': np.int32(len(emitted)),
                'min_extent_m': float(min_extent),
                'max_range_m': float(max_range),
                'ceiling_z_m': float(ceil_z),
                'ceiling_margin_m': float(ceiling_margin),
                'gt_config_hash': src_cfg.get('config_hash', 'unknown'),
                'frame_id': world_frame,
                'built_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
        for k, v in meta.items():
            f.create_dataset(f'_metadata/{k}', data=v,
                             dtype=sdt if isinstance(v, str) else None)
        for e in emitted:
            keys = np.unique(np.concatenate(e['vox'])) if e['vox'] else np.zeros(0, np.int64)
            pts = _unpack(keys, voxel)
            if e['n_emb']:
                sem = e['emb_sum'] / e['n_emb']
                sem = (sem / max(np.linalg.norm(sem), 1e-12)).astype(np.float32)
            else:
                sem = np.zeros(0, np.float32)   # no encodable observation
            dmin = min(a for a, _ in e['dist']) if e['dist'] else np.nan
            dmean = float(np.mean([b for _, b in e['dist']])) if e['dist'] else np.nan
            g = f.create_group(f'tracklets/{e["tid"]:05d}')
            g.create_dataset('tracklet_id', data=np.uint32(e['tid']))
            g.create_dataset('tracklet_geometry/header/stamp_ns',
                             data=np.int64(stamps[e['last']]))
            g.create_dataset('tracklet_geometry/header/frame_id',
                             data=world_frame, dtype=sdt)
            p = g.create_dataset('tracklet_geometry/points', data=pts,
                                 compression='gzip', compression_opts=4)
            p.attrs['fields'] = 'x,y,z'
            p.attrs['unit'] = 'm'
            p.attrs['voxel_m'] = voxel
            g.create_dataset('tracklet_semantics', data=sem)
            g.create_dataset('_metadata/point_rgb', data=voxel_rgb(e, keys),
                             compression='gzip', compression_opts=4)
            md = {'gt_object_id': np.uint32(e['oid']),
                  'color_hex': color_of.get(e['oid'], ''),
                  'first_frame': np.int64(e['first']),
                  'last_frame': np.int64(e['last']),
                  'emit_frame': np.int64(e['emit']),
                  'first_stamp_ns': np.int64(stamps[e['first']]),
                  'last_stamp_ns': np.int64(stamps[e['last']]),
                  'emit_stamp_ns': np.int64(stamps[e['emit']]),
                  'n_frames_observed': np.int32(len(e['frames'])),
                  'n_points_raw': np.int64(e['n_raw']),
                  'n_points_far': np.int64(e['n_far']),
                  'dist_min_m': np.float32(dmin),
                  'dist_mean_m': np.float32(dmean),
                  'n_obs_embedded': np.int32(e['n_emb']),
                  'flushed_at_end': np.bool_(e['flushed'])}
            for k, v in md.items():
                g.create_dataset(f'_metadata/{k}', data=v,
                                 dtype=sdt if isinstance(v, str) else None)
            rows.append([e['tid'], e['oid'], e['first'], e['last'], e['emit'],
                         len(e['frames']), len(pts), e['n_raw'], e['n_emb'],
                         int(stamps[e['first']]), int(stamps[e['last']]),
                         int(stamps[e['emit']]), int(e['flushed']),
                         e['n_far'], round(dmin, 4), round(dmean, 4)])
        _write_index(f, rows)
    _write_csv(gt, rows)
    log(f'[{gt.name}] wrote {out} ({os.path.getsize(out) / 2**20:.0f} MB, '
        f'{len(rows)} tracklets, {time.time() - t0:.0f}s)')
    return out


def ceiling_cut(z, bin_m=0.05, frac=0.05, below=0.3):
    """z threshold just below the ceiling: the highest 5 cm z-slab that is
    dense (>= frac of the densest slab) is taken as the ceiling; returns
    `below` metres under it. Percentile cuts fail here because a flat
    ceiling holds a large share of the voxels."""
    hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + bin_m, bin_m))
    dense = np.flatnonzero(hist >= frac * hist.max())
    return float(edges[dense[-1]] - below)


def ceiling_z(gt):
    """Sequence-constant ceiling height from the GT map (top dense slab).
    Fine for single-storey scenes (office); a multi-storey map gets its
    top-floor ceiling only."""
    clouds = np.load(Path(gt) / 'gt_object_clouds.npz')
    z = np.concatenate([clouds[k][:, 2] for k in clouds.files])
    return ceiling_cut(z, below=0.0)


def _thin(pts, min_extent):
    return len(pts) == 0 or bool(((pts.max(0) - pts.min(0)) <= min_extent).any())


def _on_ceiling(pts, ceil_z, margin):
    """Every point within `margin` below the ceiling (lights, sprinklers)."""
    return len(pts) > 0 and bool(pts[:, 2].min() >= ceil_z - margin)


def _drop_thin(emitted, voxel, min_extent, ceil_z, ceil_margin, log, name):
    keep, n_thin, n_ceil = [], 0, 0
    for e in emitted:
        keys = np.unique(np.concatenate(e['vox'])) if e['vox'] else np.zeros(0, np.int64)
        pts = _unpack(keys, voxel)
        if _thin(pts, min_extent):
            n_thin += 1
        elif _on_ceiling(pts, ceil_z, ceil_margin):
            n_ceil += 1
        else:
            keep.append(e)
    for k, e in enumerate(keep):
        e['tid'] = k + 1
    log(f'[{name}] dropped {n_thin} tracklets with extent <= {min_extent} m on '
        f'some axis and {n_ceil} within {ceil_margin} m of the ceiling '
        f'(z={ceil_z:.2f}), {len(keep)} kept')
    return keep


def filter_file(gt_dir, min_extent=0.04, ceiling_margin=0.3, log=print):
    """Remove tracklets whose voxel extent is <= min_extent on any axis, or
    that lie entirely within ceiling_margin of the ceiling, from an existing
    gt_tracklets.h5 (no rebuild, no CLIP); ids are renumbered 1..N in the
    existing order and index/ + the csv are rewritten."""
    gt = Path(gt_dir)
    ceil_z = ceiling_z(gt)
    src = gt / 'gt_tracklets.h5'
    tmp = gt / 'gt_tracklets.h5.tmp'
    rows = []
    with h5py.File(src, 'r') as fi, h5py.File(tmp, 'w') as fo:
        fi.copy('_metadata', fo)
        if 'min_extent_m' in fo['_metadata']:
            del fo['_metadata/min_extent_m']
        fo.create_dataset('_metadata/min_extent_m', data=float(min_extent))
        for k, v in (('ceiling_z_m', ceil_z), ('ceiling_margin_m', ceiling_margin)):
            if k in fo['_metadata']:
                del fo[f'_metadata/{k}']
            fo.create_dataset(f'_metadata/{k}', data=float(v))
        names = sorted(fi['tracklets'])
        tid = n_thin = n_ceil = 0
        for name in names:
            g = fi['tracklets'][name]
            pts = g['tracklet_geometry/points'][()]
            if _thin(pts, min_extent):
                n_thin += 1
                continue
            if _on_ceiling(pts, ceil_z, ceiling_margin):
                n_ceil += 1
                continue
            tid += 1
            fi.copy(g, fo, name=f'tracklets/{tid:05d}')
            go = fo[f'tracklets/{tid:05d}']
            del go['tracklet_id']
            go.create_dataset('tracklet_id', data=np.uint32(tid))
            md = go['_metadata']
            rows.append([tid, int(md['gt_object_id'][()]),
                         int(md['first_frame'][()]), int(md['last_frame'][()]),
                         int(md['emit_frame'][()]), int(md['n_frames_observed'][()]),
                         len(go['tracklet_geometry/points']),
                         int(md['n_points_raw'][()]), int(md['n_obs_embedded'][()]),
                         int(md['first_stamp_ns'][()]), int(md['last_stamp_ns'][()]),
                         int(md['emit_stamp_ns'][()]), int(bool(md['flushed_at_end'][()])),
                         int(md['n_points_far'][()]),
                         round(float(md['dist_min_m'][()]), 4),
                         round(float(md['dist_mean_m'][()]), 4)])
        del fo['_metadata/n_tracklets']
        fo.create_dataset('_metadata/n_tracklets', data=np.int32(tid))
        _write_index(fo, rows)
        n_in = len(names)
    os.replace(tmp, src)
    _write_csv(gt, rows)
    log(f'[{gt.name}] filter: {n_in} -> {tid} tracklets ({n_thin} extent <= '
        f'{min_extent} m, {n_ceil} within {ceiling_margin} m of ceiling z='
        f'{ceil_z:.2f}; {os.path.getsize(src) / 2**20:.0f} MB)')
    return tid


INDEX_COLS = ['tracklet_id', 'gt_object_id', 'first_frame', 'last_frame',
              'emit_frame', 'n_frames_observed', 'n_points', 'n_points_raw',
              'n_obs_embedded', 'first_stamp_ns', 'last_stamp_ns',
              'emit_stamp_ns', 'flushed_at_end', 'n_points_far',
              'dist_min_m', 'dist_mean_m']
FLOAT_COLS = {'dist_min_m', 'dist_mean_m'}


def _write_index(f, rows):
    """Columnar per-file index (same columns as the CSV, tracklet-id order)."""
    if 'index' in f:
        del f['index']
    for k, name in enumerate(INDEX_COLS):
        dt = np.float32 if name in FLOAT_COLS else np.int64
        f.create_dataset(f'index/{name}', data=np.array([r[k] for r in rows], dt))


def _write_csv(gt, rows):
    with open(gt / 'gt_tracklets_index.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(INDEX_COLS)
        w.writerows(rows)


def upgrade_file(gt_dir, log=print):
    """Bring an existing gt_tracklets.h5 to the current layout without
    re-running CLIP: empty semantics for n_obs_embedded==0, per-tracklet
    color_hex/last_stamp_ns, file-level index/ and engine metadata."""
    import hashlib
    gt = Path(gt_dir).expanduser()
    color_of = {}
    with open(gt / 'gt_objects.csv') as fh:
        color_of = {int(r['gt_object_id']): r['color_hex'] for r in csv.DictReader(fh)}
    sdt = h5py.string_dtype()
    rows, n_empty = [], 0
    with h5py.File(gt / 'gt_tracklets.h5', 'r+') as f:
        for name in sorted(f['tracklets']):
            g = f['tracklets'][name]
            md = g['_metadata']
            if int(md['n_obs_embedded'][()]) == 0 and g['tracklet_semantics'].shape != (0,):
                del g['tracklet_semantics']
                g.create_dataset('tracklet_semantics', data=np.zeros(0, np.float32))
                n_empty += 1
            oid = int(md['gt_object_id'][()])
            if 'color_hex' not in md:
                md.create_dataset('color_hex', data=color_of.get(oid, ''), dtype=sdt)
            if 'last_stamp_ns' not in md:
                md.create_dataset('last_stamp_ns',
                                  data=np.int64(g['tracklet_geometry/header/stamp_ns'][()]))
            rows.append([int(name), oid, int(md['first_frame'][()]), int(md['last_frame'][()]),
                         int(md['emit_frame'][()]), int(md['n_frames_observed'][()]),
                         int(g['tracklet_geometry/points'].shape[0]), int(md['n_points_raw'][()]),
                         int(md['n_obs_embedded'][()]), int(md['first_stamp_ns'][()]),
                         int(md['last_stamp_ns'][()]), int(md['emit_stamp_ns'][()]),
                         int(bool(md['flushed_at_end'][()]))])
        _write_index(f, rows)
        fm = f['_metadata']
        if 'embedding_engine' not in fm:
            eng = Path(DEFAULT_MODEL_DIR) / 'clip_vit_b32_visual_pooled_value_fp16.engine'
            fm.create_dataset('embedding_engine', data=eng.name, dtype=sdt)
            sha = hashlib.sha1(eng.read_bytes()).hexdigest()[:12] if eng.exists() else 'unknown'
            fm.create_dataset('embedding_engine_sha1', data=sha, dtype=sdt)
    log(f'[{gt.name}] upgraded gt_tracklets.h5: {len(rows)} tracklets, '
        f'{n_empty} semantics emptied, index/ written')


# ---------------------------------------------------------------- read ----

def _scalar(ds):
    v = ds[()]
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray) and v.shape != ():
        return v                      # array-valued metadata (e.g. point_rgb)
    return v.item() if hasattr(v, 'item') else v


def load_tracklets(path, ids=None, with_points=True):
    """gt_tracklets.h5 -> (file_metadata, {tid: {tracklet_id,
    tracklet_geometry{header, points}, tracklet_semantics, _metadata}})."""
    out = {}
    with h5py.File(path, 'r') as f:
        meta = {k: _scalar(f['_metadata'][k]) for k in f['_metadata']}
        for name in f['tracklets']:
            tid = int(name)
            if ids is not None and tid not in ids:
                continue
            g = f['tracklets'][name]
            out[tid] = {
                'tracklet_id': _scalar(g['tracklet_id']),
                'tracklet_geometry': {
                    'header': {'stamp_ns': _scalar(g['tracklet_geometry/header/stamp_ns']),
                               'frame_id': _scalar(g['tracklet_geometry/header/frame_id'])},
                    'points': g['tracklet_geometry/points'][()] if with_points else None},
                'tracklet_semantics': g['tracklet_semantics'][()],
                '_metadata': {m: _scalar(g['_metadata'][m]) for m in g['_metadata']}}
    return meta, out


def load_index(path):
    """Columnar index/ group -> {column: int64 array} (tracklet-id order)."""
    with h5py.File(path, 'r') as f:
        return {k: f['index'][k][()] for k in f['index']}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--gt', required=True, help='GT dir for one sequence')
    ap.add_argument('--dataset', default='',
                    help='uHumans2 sequence dir (rgb for CLIP); not needed '
                         'for --upgrade/--filter')
    ap.add_argument('--gap', type=int, default=5,
                    help='max unobserved frames inside one tracklet')
    ap.add_argument('--voxel', type=float, default=0.02)
    ap.add_argument('--no-embed', action='store_true',
                    help='skip CLIP (tracklet_semantics = zeros)')
    ap.add_argument('--backend', default='tensorrt', choices=['tensorrt', 'torch'])
    ap.add_argument('--model-dir', default=DEFAULT_MODEL_DIR)
    ap.add_argument('--max-frames', type=int, default=0, help='debug')
    ap.add_argument('--recolor', action='store_true',
                    help='only (re)compute _metadata/point_rgb in an existing '
                         'gt_tracklets.h5 (no CLIP, no geometry rewrite)')
    ap.add_argument('--max-range', type=float, default=5.0,
                    help='drop observed points farther than this from the '
                         'camera (m); tracklets left empty are dropped')
    ap.add_argument('--min-extent', type=float, default=0.04,
                    help='drop tracklets whose voxel extent is <= this on '
                         'any axis (single voxel sheets/lines)')
    ap.add_argument('--ceiling-margin', type=float, default=0.3,
                    help='drop tracklets lying entirely within this distance '
                         'below the ceiling (sequence-constant, from the GT map)')
    ap.add_argument('--filter', action='store_true',
                    help='only apply --min-extent to an existing '
                         'gt_tracklets.h5 (renumbers ids; no rebuild, no CLIP)')
    ap.add_argument('--upgrade', action='store_true',
                    help='only bring an existing gt_tracklets.h5 to the '
                         'current layout (no rebuild, no CLIP)')
    a = ap.parse_args()
    if a.upgrade:
        upgrade_file(a.gt)
        return
    if a.filter:
        filter_file(a.gt, a.min_extent, a.ceiling_margin)
        return
    build(a.gt, a.dataset, gap=a.gap, voxel=a.voxel, embed=not a.no_embed,
          backend=a.backend, model_dir=a.model_dir, max_frames=a.max_frames,
          recolor_only=a.recolor, min_extent=a.min_extent,
          max_range=a.max_range, ceiling_margin=a.ceiling_margin)


if __name__ == '__main__':
    main()
