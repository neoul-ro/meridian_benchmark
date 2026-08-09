"""Build benchmark GT artifacts from one uHumans2 unpacked sequence.

Identity model: a uHumans2 seg colour is a *prefab/material type*, not an
instance (the same colour appears on every copy of a prop, and on reflections
seen in mirrors / through glass). GT object identity is therefore derived by
decomposing each colour into 3D instances: per-frame 2D blobs of one colour
are clustered by world-space voxel overlap, so spatially separate copies
become separate gt_object_ids and mirror/glass phantoms separate from the
real object into their own (self-consistent) instances.

Passes:
  A  instance discovery — per frame, per colour: connected 2D blobs ->
     world-space voxels -> assign to / create / merge colour instances
  B  artifacts — deterministic re-extraction of the same blobs, labelled with
     final gt_object_ids: remap PNGs, frame clouds, observation table

Outputs (under --out/<sequence_name>/):
  config.json            builder config + provenance
  registry.json          per-object colour/instance stats, per-colour counts
  frames.csv             frame_idx, stamp_ns, source paths
  gt_poses.csv           frame_idx, stamp_ns, world_T_left_cam (t + q xyzw)
  gt_seg/%06d.png        mono8 frame-local segment_id (0 = background)
  gt_frame_clouds/%06d.npz  world-frame strided points per segment
  segments.csv           one row per (frame, gt object) observation
  gt_objects.csv         one row per promoted GT object (= colour instance)
  gt_object_clouds.npz   voxel-downsampled accumulated world cloud per object
  gt_embeddings.npz      per-object embedding; shared across instances of the
                         same colour (semantics identify the type, not the copy)
"""

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from .uhumans2 import UH2Sequence, unproject, unpack_rgb

VOXEL_HALF_RANGE = 8192   # voxel index range per axis (14 bits, signed)
THETA_ASSIGN = 0.15       # blob->instance overlap fraction to assign
THETA_MERGE = 0.30        # two instances both above this get merged


def is_dynamic_object(gt_object_id, seq):
    # TODO(dynamic): 01h/02h humans should be flagged via tf/tf.csv agent
    # frames. Unimplemented by decision - all objects are treated as static
    # and only human-free sequences are scored.
    return False


def _pack_voxels(pts, voxel):
    idx = np.floor(pts / voxel).astype(np.int64)
    np.clip(idx, -VOXEL_HALF_RANGE, VOXEL_HALF_RANGE - 1, out=idx)
    idx += VOXEL_HALF_RANGE
    return (idx[:, 0] << 28) | (idx[:, 1] << 14) | idx[:, 2]


def _unpack_voxels(packed, voxel):
    ix = (packed >> 28) & 0x3FFF
    iy = (packed >> 14) & 0x3FFF
    iz = packed & 0x3FFF
    idx = np.stack([ix, iy, iz], axis=1).astype(np.float64) - VOXEL_HALF_RANGE
    return ((idx + 0.5) * voxel).astype(np.float32)


_NEIGHBOR_OFFSETS = np.array(
    [0, 1 << 28, -(1 << 28), 1 << 14, -(1 << 14), 1, -1], dtype=np.int64)


def _overlap_fraction(blob_vox, inst_vox_sorted):
    """Fraction of blob voxels that are in the instance set or 6-adjacent."""
    if len(blob_vox) == 0 or len(inst_vox_sorted) == 0:
        return 0.0
    hit = np.zeros(len(blob_vox), dtype=bool)
    for off in _NEIGHBOR_OFFSETS:
        q = blob_vox + off
        idx = np.searchsorted(inst_vox_sorted, q)
        idx[idx == len(inst_vox_sorted)] = 0
        hit |= inst_vox_sorted[idx] == q
    return float(hit.mean()) if len(blob_vox) else 0.0


def _nn_overlap(small_vox, big_vox, voxel, sample=3000):
    """Fraction of (sampled) small-cloud voxel centers within 1.5 voxel of the
    big cloud — the same criterion the verifier's instance-separation check
    uses, so build-time merging and verification cannot disagree."""
    from scipy.spatial import cKDTree
    pa = _unpack_voxels(small_vox, voxel)
    pb = _unpack_voxels(big_vox, voxel)
    if len(pa) > sample:
        pa = pa[::len(pa) // sample + 1]
    d, _ = cKDTree(pb).query(pa, k=1)
    return float((d < 1.5 * voxel).mean())


def extract_blobs(packed, depth, K, R_wc, t_wc, stride):
    """Per-colour 2D connected components with world-space strided points.

    Returns (colors, labels_by_color, blobs) where blobs is a list of dicts
    ordered deterministically (colour asc, then blob label asc).
    labels_by_color[c] = (x0, y0, labels_array) for pixel-level lookup.
    """
    pts_cam, (pv, pu) = unproject(depth, K, stride=stride)
    pts_world = pts_cam @ R_wc.T.astype(np.float32) + t_wc.astype(np.float32)
    pt_color = packed[pv, pu]

    colors = np.unique(packed)
    labels_by_color = {}
    blobs = []
    for c in colors.tolist():
        ys, xs = np.where(packed == c)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        mask = (packed[y0:y1, x0:x1] == c).astype(np.uint8)
        n_lbl, labels = cv2.connectedComponents(mask, connectivity=8)
        labels_by_color[c] = (x0, y0, labels)

        sel = pt_color == c
        if sel.any():
            bl = labels[pv[sel] - y0, pu[sel] - x0]
            pts_c = pts_world[sel]
            depth_c = pts_cam[sel][:, 2]
        else:
            bl = np.zeros(0, int)
            pts_c = np.zeros((0, 3), np.float32)
            depth_c = np.zeros(0, np.float32)

        counts = np.bincount(labels[mask > 0], minlength=n_lbl)
        for li in range(1, n_lbl):
            in_b = bl == li
            blobs.append({'color': c, 'local_id': li,
                          'n_px': int(counts[li]),
                          'pts': pts_c[in_b],
                          'mean_depth': float(depth_c[in_b].mean())
                          if in_b.any() else 0.0})
    return colors, labels_by_color, blobs


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, a):
        while self.p.setdefault(a, a) != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def discover_instances(seq, K, R_wc, t_wc, cfg, log):
    """Pass A. Returns (assign, uf, inst_vox, inst_color, px_per_frame)."""
    voxel = cfg['voxel']
    inst_vox = {}        # prov instance id -> sorted voxel array
    inst_dirty = {}      # unsorted pending chunks
    inst_color = {}
    by_color = {}        # colour -> [prov instance ids]
    assign = {}          # (frame, colour, local_id) -> prov instance id
    px_per_frame = {}    # prov instance id -> {frame: n_px}
    uf = _UF()
    next_id = [0]
    t0 = time.time()

    def compact(iid):
        if inst_dirty.get(iid):
            inst_vox[iid] = np.unique(np.concatenate(
                [inst_vox[iid]] + inst_dirty[iid]))
            inst_dirty[iid] = []

    for i in range(seq.n_frames):
        packed = seq.seg_packed(i)
        depth = seq.depth(i)
        _, _, blobs = extract_blobs(packed, depth, K, R_wc[i], t_wc[i],
                                    cfg['stride'])
        for b in sorted(blobs, key=lambda b: (-b['n_px'], b['color'],
                                              b['local_id'])):
            c = b['color']
            vox = np.unique(_pack_voxels(b['pts'], voxel)) if len(b['pts']) \
                else np.zeros(0, np.int64)
            cand = by_color.setdefault(c, [])
            best, best_f, merged = None, 0.0, []
            if len(vox):
                for iid in cand:
                    compact(iid)
                    f = _overlap_fraction(vox, inst_vox[iid])
                    if f > best_f:
                        best, best_f = iid, f
                    if f >= THETA_MERGE:
                        merged.append(iid)
            if best is not None and best_f >= THETA_ASSIGN:
                iid = best
                for other in merged:
                    uf.union(iid, other)
            elif best is not None and len(vox) < cfg['min_new_vox']:
                iid = best  # tiny sliver: attach to closest-overlap instance
            elif len(vox) < cfg['min_new_vox']:
                # sliver with no spatial anchor: leave unlabelled instead of
                # seeding an instance (sliver instances never promote but
                # bloat the per-colour candidate scan)
                assign[(i, c, b['local_id'])] = -1
                continue
            else:
                iid = next_id[0]
                next_id[0] += 1
                inst_vox[iid] = np.zeros(0, np.int64)
                inst_dirty[iid] = []
                inst_color[iid] = c
                cand.append(iid)
            if len(vox):
                inst_dirty.setdefault(iid, []).append(vox)
                if sum(len(x) for x in inst_dirty[iid]) > 4 * max(
                        1, len(inst_vox[iid])):
                    compact(iid)
            assign[(i, c, b['local_id'])] = iid
            px_per_frame.setdefault(iid, {})
            px_per_frame[iid][i] = px_per_frame[iid].get(i, 0) + b['n_px']

        if (i + 1) % 200 == 0:
            log(f'  discover {i + 1}/{seq.n_frames} '
                f'instances={len(inst_vox)} ({time.time() - t0:.0f}s)')

    for iid in list(inst_vox):
        compact(iid)
    return assign, uf, inst_vox, inst_color, px_per_frame


def build(seq_root, out_root, cfg, log=print):
    seq = UH2Sequence(seq_root)
    name = Path(seq_root).name
    out = Path(out_root) / name
    (out / 'gt_seg').mkdir(parents=True, exist_ok=True)
    (out / 'gt_frame_clouds').mkdir(exist_ok=True)

    K = seq.camera_info('left_cam')['K']
    t_wc, q_wc = seq.camera_poses()
    from scipy.spatial.transform import Rotation
    R_wc = Rotation.from_quat(q_wc).as_matrix()

    log(f'[{name}] pass A: instance discovery')
    assign, uf, inst_vox, inst_color, px_per_frame = discover_instances(
        seq, K, R_wc, t_wc, cfg, log)

    # resolve union-find -> final instances
    final_vox, final_color, final_px = {}, {}, {}
    for iid, vox in inst_vox.items():
        r = uf.find(iid)
        final_color[r] = inst_color[r]
        if r in final_vox:
            final_vox[r] = np.unique(np.concatenate([final_vox[r], vox]))
        else:
            final_vox[r] = vox
        acc = final_px.setdefault(r, {})
        for f, n in px_per_frame.get(iid, {}).items():
            acc[f] = acc.get(f, 0) + n

    # post-pass merge to fixpoint: same-colour instances occupying the same
    # space are one object that escaped incremental merging (occlusion-split
    # blobs each seeded an instance and later observations fed only one side)
    uf2 = _UF()
    n_merged = 0
    for _round in range(5):
        cur_vox = {}
        for r in final_vox:
            rr = uf2.find(r)
            cur_vox.setdefault(rr, []).append(final_vox[r])
        cur_vox = {r: np.unique(np.concatenate(v)) if len(v) > 1 else v[0]
                   for r, v in cur_vox.items()}
        by_color = {}
        for r in cur_vox:
            by_color.setdefault(final_color[uf2.find(r)]
                                if uf2.find(r) in final_color
                                else final_color[r], []).append(r)
        changed = False
        for c, roots in by_color.items():
            roots.sort(key=lambda r: -len(cur_vox[r]))
            for ai in range(len(roots)):
                for bi in range(ai + 1, len(roots)):
                    a, b = roots[ai], roots[bi]
                    if uf2.find(a) == uf2.find(b):
                        continue
                    small, big = (a, b) if len(cur_vox[a]) <= len(cur_vox[b]) \
                        else (b, a)
                    if not len(cur_vox[small]):
                        continue
                    if _nn_overlap(cur_vox[small], cur_vox[big],
                                   cfg['voxel']) >= \
                            cfg['post_merge_overlap']:
                        uf2.union(a, b)
                        changed = True
                        n_merged += 1
        if not changed:
            break
    if n_merged:
        log(f'[{name}] post-merge: {n_merged} instance pairs merged')
        merged_vox, merged_px, merged_color = {}, {}, {}
        for r in final_vox:
            rr = uf2.find(r)
            merged_color[rr] = final_color[r]
            if rr in merged_vox:
                merged_vox[rr] = np.unique(np.concatenate(
                    [merged_vox[rr], final_vox[r]]))
            else:
                merged_vox[rr] = final_vox[r]
            acc = merged_px.setdefault(rr, {})
            for f, n in final_px[r].items():
                acc[f] = acc.get(f, 0) + n
        final_vox, final_px, final_color = merged_vox, merged_px, merged_color

    # promotion + gt_object_id assignment
    promoted = []
    for r, acc in final_px.items():
        good = sum(1 for n in acc.values() if n >= cfg['min_px'])
        if good >= cfg['min_frames']:
            promoted.append((min(acc), final_color[r], r))
    promoted.sort()
    root_to_oid = {r: k + 1 for k, (_, _, r) in enumerate(promoted)}
    n_obj = len(root_to_oid)
    n_colors_prom = len({final_color[r] for r in root_to_oid})
    log(f'[{name}] instances={len(final_vox)} promoted={n_obj} '
        f'(colours with promoted instance: {n_colors_prom})')

    def oid_of(frame, color, local_id):
        iid = assign[(frame, color, local_id)]
        if iid < 0:
            return 0
        return root_to_oid.get(uf2.find(uf.find(iid)), 0)

    log(f'[{name}] pass B: per-frame artifacts')
    seg_rows = []
    t0 = time.time()
    for i in range(seq.n_frames):
        packed = seq.seg_packed(i)
        depth = seq.depth(i)
        colors, labels_by_color, blobs = extract_blobs(
            packed, depth, K, R_wc[i], t_wc[i], cfg['stride'])

        # group blobs by final object
        per_obj = {}
        for b in blobs:
            oid = oid_of(i, b['color'], b['local_id'])
            if oid == 0:
                continue
            per_obj.setdefault(oid, []).append(b)

        sid_of_obj = {oid: sid for sid, oid in
                      enumerate(sorted(per_obj), start=1)}
        if len(sid_of_obj) > 255:
            raise RuntimeError(f'frame {i}: {len(sid_of_obj)} segments > 255')

        seg_img = np.zeros(packed.shape, dtype=np.uint8)
        cloud_pts, offsets, sids, oids = [], [0], [], []
        for oid, obs in sorted(per_obj.items()):
            sid = sid_of_obj[oid]
            for b in obs:
                x0, y0, labels = labels_by_color[b['color']]
                m = labels == b['local_id']
                seg_img[y0:y0 + labels.shape[0],
                        x0:x0 + labels.shape[1]][m] = sid
            pts = np.concatenate([b['pts'] for b in obs]) if obs else \
                np.zeros((0, 3), np.float32)
            n_px = sum(b['n_px'] for b in obs)
            wsum = sum(b['mean_depth'] * len(b['pts']) for b in obs)
            npts = sum(len(b['pts']) for b in obs)
            seg_rows.append((i, int(seq.stamps_ns[i]), sid, oid, n_px,
                             npts, round(wsum / npts, 4) if npts else 0.0,
                             int(is_dynamic_object(oid, seq))))
            sids.append(sid)
            oids.append(oid)
            cloud_pts.append(pts)
            offsets.append(offsets[-1] + len(pts))

        cv2.imwrite(str(out / 'gt_seg' / f'{i:06d}.png'), seg_img)
        np.savez_compressed(
            out / 'gt_frame_clouds' / f'{i:06d}.npz',
            stamp_ns=np.int64(seq.stamps_ns[i]),
            segment_ids=np.array(sids, dtype=np.uint8),
            gt_object_ids=np.array(oids, dtype=np.uint32),
            offsets=np.array(offsets, dtype=np.int64),
            points=np.concatenate(cloud_pts).astype(np.float32)
            if cloud_pts else np.zeros((0, 3), np.float32))
        if (i + 1) % 200 == 0:
            log(f'  artifacts {i + 1}/{seq.n_frames} ({time.time() - t0:.0f}s)')

    # ---------------- aggregate outputs ----------------

    log(f'[{name}] aggregating')
    voxel = cfg['voxel']
    inst_index_within_color = {}
    obj_rows = []
    clouds = {}
    for _, c, r in promoted:
        oid = root_to_oid[r]
        k = inst_index_within_color.get(c, 0)
        inst_index_within_color[c] = k + 1
        pts = _unpack_voxels(final_vox[r], voxel)
        clouds[f'obj_{oid:05d}'] = pts
        acc = final_px[r]
        centroid = pts.mean(axis=0) if len(pts) else np.zeros(3)
        amin = pts.min(axis=0) if len(pts) else np.zeros(3)
        amax = pts.max(axis=0) if len(pts) else np.zeros(3)
        rgb = unpack_rgb(np.uint32(c))
        obj_rows.append(
            (oid, f'{c:06x}', k, int(rgb[0]), int(rgb[1]), int(rgb[2]),
             min(acc), max(acc), len(acc),
             sum(1 for n in acc.values() if n >= cfg['min_px']),
             max(acc.values()), sum(acc.values()), len(pts),
             *np.round(centroid, 4), *np.round(amin, 4), *np.round(amax, 4)))

    np.savez_compressed(out / 'gt_object_clouds.npz', **clouds)

    with open(out / 'gt_objects.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['gt_object_id', 'color_hex', 'instance_idx', 'r', 'g', 'b',
                    'first_frame', 'last_frame', 'n_frames_seen',
                    'n_frames_promoting', 'max_px', 'total_px', 'n_voxels',
                    'centroid_x', 'centroid_y', 'centroid_z',
                    'aabb_min_x', 'aabb_min_y', 'aabb_min_z',
                    'aabb_max_x', 'aabb_max_y', 'aabb_max_z'])
        w.writerows(obj_rows)

    with open(out / 'segments.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame_idx', 'stamp_ns', 'segment_id', 'gt_object_id',
                    'n_pixels', 'n_points_strided', 'mean_depth_m',
                    'is_dynamic'])
        w.writerows(seg_rows)

    with open(out / 'frames.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame_idx', 'stamp_ns', 'rgb', 'depth', 'seg'])
        for i in range(seq.n_frames):
            w.writerow([i, int(seq.stamps_ns[i]), seq.rgb_files[i],
                        seq.depth_files[i], seq.seg_files[i]])

    # /pose publishes the base pose (world_T_base); the camera pose is what
    # all world-frame geometry in this GT was built with (internal + eval)
    t_wb, q_wb = seq.camera_poses(cam=None)
    with open(out / 'gt_poses.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame_idx', 'stamp_ns',
                    'base_tx', 'base_ty', 'base_tz',
                    'base_qx', 'base_qy', 'base_qz', 'base_qw',
                    'cam_tx', 'cam_ty', 'cam_tz',
                    'cam_qx', 'cam_qy', 'cam_qz', 'cam_qw'])
        for i in range(seq.n_frames):
            w.writerow([i, int(seq.stamps_ns[i]),
                        *np.round(t_wb[i], 6), *np.round(q_wb[i], 8),
                        *np.round(t_wc[i], 6), *np.round(q_wc[i], 8)])

    # embeddings shared per colour: semantics identify the type, not the copy
    colors_sorted = sorted({c for _, c, _ in promoted})
    D = cfg['embed_dim']
    rng = np.random.default_rng(cfg['embed_seed'])
    if len(colors_sorted) <= D:
        color_emb = np.eye(D, dtype=np.float32)[:len(colors_sorted)]
    else:
        color_emb = rng.standard_normal(
            (len(colors_sorted), D)).astype(np.float32)
        color_emb /= np.linalg.norm(color_emb, axis=1, keepdims=True)
    color_row = {c: k for k, c in enumerate(colors_sorted)}
    emb = np.stack([color_emb[color_row[c]] for _, c, r in promoted]) \
        if promoted else np.zeros((0, D), np.float32)
    np.savez_compressed(out / 'gt_embeddings.npz',
                        gt_object_ids=np.arange(1, n_obj + 1,
                                                dtype=np.uint32),
                        embeddings=emb,
                        embedding_model_id=np.str_('bench/gt-typecolor-v1'))

    per_color_counts = {}
    for r in final_vox:
        c = final_color[r]
        per_color_counts.setdefault(f'{c:06x}', [0, 0])[0] += 1
    for r in root_to_oid:
        per_color_counts[f'{final_color[r]:06x}'][1] += 1
    with open(out / 'registry.json', 'w') as f:
        json.dump({
            'identity_model': 'color x 3D-instance (colour = prefab type)',
            'objects': {str(root_to_oid[r]):
                        {'color_hex': f'{final_color[r]:06x}'}
                        for r in root_to_oid},
            'per_color_instances': per_color_counts}, f)

    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        sha = 'unknown'
    cfg_hash = hashlib.sha1(
        json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    with open(out / 'config.json', 'w') as f:
        json.dump({'config': cfg, 'config_hash': cfg_hash,
                   'builder_git': sha, 'dataset': str(seq_root),
                   'n_frames': seq.n_frames, 'n_objects': n_obj,
                   'n_instances_total': len(final_vox),
                   'built_at': time.strftime('%Y-%m-%dT%H:%M:%S')},
                  f, indent=1)

    log(f'[{name}] done: {n_obj} objects, {len(seg_rows)} observations '
        f'-> {out}')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dataset', required=True,
                    help='uHumans2 unpacked sequence dir')
    ap.add_argument('--out', required=True, help='GT output root')
    ap.add_argument('--stride', type=int, default=4)
    ap.add_argument('--voxel', type=float, default=0.02)
    ap.add_argument('--min-px', type=int, default=100)
    ap.add_argument('--min-frames', type=int, default=3)
    ap.add_argument('--min-new-vox', type=int, default=5)
    ap.add_argument('--post-merge-overlap', type=float, default=0.15)
    ap.add_argument('--embed-dim', type=int, default=512)
    ap.add_argument('--embed-seed', type=int, default=0)
    a = ap.parse_args()
    cfg = {'stride': a.stride, 'voxel': a.voxel, 'min_px': a.min_px,
           'min_frames': a.min_frames, 'min_new_vox': a.min_new_vox,
           'post_merge_overlap': a.post_merge_overlap,
           'embed_dim': a.embed_dim, 'embed_seed': a.embed_seed}
    build(a.dataset, a.out, cfg)


if __name__ == '__main__':
    main()
