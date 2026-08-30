"""Build benchmark GT artifacts from one uHumans2 unpacked sequence.

Identity model: a uHumans2 seg colour is a *prefab/material type*, not an
instance (the same colour appears on every copy of a prop, and on reflections
seen in mirrors / through glass). GT object identity is therefore derived by
decomposing each colour into 3D instances: per-frame 2D blobs of one colour
are clustered by world-space voxel overlap, so spatially separate copies
become separate gt_object_ids and mirror/glass phantoms separate from the
real object into their own (self-consistent) instances.

Passes:
  A  instance discovery — per frame, per colour: connected 2D blobs, each
     split by 3D connectivity (copies that only overlap in the image stay
     apart) -> world-space voxels -> assign to / create / merge instances
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
                         (is_far = no strided point within near_m)
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
SPLIT_GAP_FACTOR = 1.5    # 3D link cell = this x strided sampling gap at depth


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


def _nn_overlap(small_vox, big_vox, voxel, sample=3000, tree=None):
    """Fraction of (sampled) small-cloud voxel centers within 1.5 voxel of the
    big cloud — the same criterion the verifier's instance-separation check
    uses, so build-time merging and verification cannot disagree."""
    from scipy.spatial import cKDTree
    pa = _unpack_voxels(small_vox, voxel)
    if len(pa) > sample:
        pa = pa[::len(pa) // sample + 1]
    if tree is None:
        tree = cKDTree(_unpack_voxels(big_vox, voxel))
    d, _ = tree.query(pa, k=1)
    return float((d < 1.5 * voxel).mean())


def _vox_bbox(vox):
    idx = np.stack([(vox >> 28) & 0x3FFF, (vox >> 14) & 0x3FFF,
                    vox & 0x3FFF], 1)
    return idx.min(0), idx.max(0)


def post_merge(final_vox, final_color, voxel, threshold, rounds=50,
               sample=3000, k=20):
    """Merge same-colour instances occupying the same space to a fixpoint
    (occlusion-split blobs each seeded an instance and later observations
    fed only one side). Returns (union-find over roots, n_merged).

    Per colour and round, one KD-tree over every voxel centre of that colour
    (labelled by root); each root's sampled voxels are queried (k=20 covers
    every voxel centre within 1.5 voxel) and a sample counts for every other
    root it is near, so the per-pair fraction equals the pairwise test the
    verifier runs; a root is merged into the other root with the largest
    fraction >= `threshold`. Smaller roots are resolved first so fragments
    join the dominant instance; rounds repeat until nothing changes."""
    from scipy.spatial import cKDTree
    uf2 = _UF()
    n_merged = 0
    for _round in range(rounds):
        cur_vox = {}
        for r in final_vox:
            cur_vox.setdefault(uf2.find(r), []).append(final_vox[r])
        cur_vox = {r: np.unique(np.concatenate(v)) if len(v) > 1 else v[0]
                   for r, v in cur_vox.items() if sum(map(len, v))}
        by_color = {}
        for r in cur_vox:
            by_color.setdefault(final_color[r], []).append(r)
        changed = False
        for c, roots in by_color.items():
            if len(roots) < 2:
                continue
            roots.sort(key=lambda r: (-len(cur_vox[r]), r))
            pts = np.concatenate([_unpack_voxels(cur_vox[r], voxel) for r in roots])
            lab = np.repeat(np.arange(len(roots)),
                            [len(cur_vox[r]) for r in roots])
            tree = cKDTree(pts)
            starts = np.r_[0, np.cumsum([len(cur_vox[r]) for r in roots])]
            for ri in range(len(roots) - 1, 0, -1):     # smallest first
                q = pts[starts[ri]:starts[ri + 1]]
                if len(q) > sample:
                    q = q[::len(q) // sample + 1]
                d, nn = tree.query(q, k=k, distance_upper_bound=1.5 * voxel)
                ok = np.isfinite(d)
                other = np.where(ok, lab[np.minimum(nn, len(lab) - 1)], -1)
                other[other == ri] = -1
                # each sample counts once per distinct other root it touches
                si = np.repeat(np.arange(len(q)), other.shape[1])
                keep = other.ravel() >= 0
                pair = np.unique(si[keep] * len(roots) + other.ravel()[keep])
                if not len(pair):
                    continue
                cnt = np.bincount(pair % len(roots), minlength=len(roots))
                best = int(np.argmax(cnt))
                if cnt[best] / len(q) >= threshold:
                    uf2.union(roots[best], roots[ri])
                    changed = True
                    n_merged += 1
        if not changed:
            break
    return uf2, n_merged


def _label_grid(pts, cell):
    from scipy import ndimage
    idx = np.floor((pts - pts.min(0)) / cell).astype(np.int64)
    shape = idx.max(0) + 1
    grid = np.zeros(shape, bool)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    lab, _ = ndimage.label(grid, structure=np.ones((3, 3, 3), int))
    return lab[idx[:, 0], idx[:, 1], idx[:, 2]]


def _split_3d(pts, radius):
    """Connected components of world points with a per-point link scale:
    `radius[i]` is the grid cell at which point i may join a neighbour
    (>= the strided sampling gap at its depth). Near points only link at
    the finest cell, so copies 2+ cells apart stay separate, while the
    sparse far part of the same blob is not shredded into fragments.
    Returns component label per point (0..k-1), largest component first."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(pts)
    rows, cols = [], []
    cell = float(radius.min())
    rmax = float(radius.max())
    while True:
        elig = np.flatnonzero(radius >= cell / 2)
        if len(elig) > 1:
            lab = _label_grid(pts[elig], cell)
            order = np.argsort(lab, kind='stable')
            e, l = elig[order], lab[order]
            first = np.flatnonzero(np.r_[True, l[1:] != l[:-1]])
            rep = np.repeat(e[first], np.diff(np.r_[first, len(l)]))
            rows.append(rep)
            cols.append(e)
        if cell >= rmax:
            break
        cell *= 2
    if not rows:
        return np.zeros(n, np.int64), 1
    r = np.concatenate(rows); c = np.concatenate(cols)
    g = coo_matrix((np.ones(len(r), np.int8), (r, c)), shape=(n, n))
    k, comp = connected_components(g, directed=False)
    if k == 1:
        return np.zeros(n, np.int64), 1
    # deterministic order: size desc, then smallest lexicographic point
    key = np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))
    rank = np.empty(n, np.int64); rank[key] = np.arange(n)
    sizes = np.bincount(comp, minlength=k)
    minrank = np.full(k, n, np.int64)
    np.minimum.at(minrank, comp, rank)
    order = np.lexsort((minrank, -sizes))
    remap = np.empty(k, np.int64); remap[order] = np.arange(k)
    return remap[comp], k


def extract_blobs(packed, depth, K, R_wc, t_wc, stride, split_cell=0.10):
    """Per-colour identity blobs with world-space strided points.

    A 2D connected component of one colour is split further by 3D
    connectivity (`_split_3d`), so same-prefab copies that merely overlap in
    the image (a row of armchairs) become separate blobs; the link scale
    grows with each point's depth so strided sampling gaps never split one
    object.
    Every pixel of the 2D component is assigned to the sub-blob of its
    nearest strided pixel.

    Returns (colors, labels_by_color, blobs) where blobs is a list of dicts
    ordered deterministically (colour asc, then blob label asc).
    labels_by_color[c] = (x0, y0, labels_array) for pixel-level lookup.
    """
    pts_cam, (pv, pu) = unproject(depth, K, stride=stride)
    pts_world = pts_cam @ R_wc.T.astype(np.float32) + t_wc.astype(np.float32)
    pt_color = packed[pv, pu]
    fx = float(K[0, 0])

    colors = np.unique(packed)
    labels_by_color = {}
    blobs = []
    for c in colors.tolist():
        ys, xs = np.where(packed == c)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        mask = (packed[y0:y1, x0:x1] == c).astype(np.uint8)
        n_lbl, labels2d = cv2.connectedComponents(mask, connectivity=8)

        sel = pt_color == c
        if sel.any():
            bl = labels2d[pv[sel] - y0, pu[sel] - x0]
            pts_c = pts_world[sel]
            depth_c = pts_cam[sel][:, 2]
            pv_c, pu_c = pv[sel], pu[sel]
        else:
            bl = np.zeros(0, int)
            pts_c = np.zeros((0, 3), np.float32)
            depth_c = np.zeros(0, np.float32)
            pv_c = pu_c = np.zeros(0, int)

        labels = np.zeros_like(labels2d)
        next_local = 1
        for li in range(1, n_lbl):
            in_b = bl == li
            pix = labels2d == li
            if in_b.sum() < 2:
                labels[pix] = next_local
                blobs.append({'color': c, 'local_id': next_local,
                              'n_px': int(pix.sum()), 'pts': pts_c[in_b],
                              'depths': depth_c[in_b],
                              'pix': (pv_c[in_b], pu_c[in_b]),
                              'mean_depth': float(depth_c[in_b].mean())
                              if in_b.any() else 0.0})
                next_local += 1
                continue
            bp, bd = pts_c[in_b], depth_c[in_b]
            radius = np.maximum(split_cell, SPLIT_GAP_FACTOR * stride * bd / fx)
            comp, n_sub = _split_3d(bp, radius)
            yy, xx = np.nonzero(pix)
            if n_sub == 1:
                pix_lab = np.zeros(len(yy), np.int64)
            else:
                # every pixel of the 2D component -> sub-blob of its nearest
                # strided pixel (2D; seeds are <= stride/2 px away)
                seeds = np.full(labels2d.shape, 255, np.uint8)
                sy, sx = pv_c[in_b] - y0, pu_c[in_b] - x0
                seeds[sy, sx] = 0
                _, lbl = cv2.distanceTransformWithLabels(
                    seeds, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
                seed_comp = np.zeros(lbl.max() + 1, np.int64)
                seed_comp[lbl[sy, sx]] = comp
                pix_lab = seed_comp[lbl[yy, xx]]
            for k in range(n_sub):
                m_pts = comp == k
                m_pix = pix_lab == k
                labels[yy[m_pix], xx[m_pix]] = next_local
                blobs.append({'color': c, 'local_id': next_local,
                              'n_px': int(m_pix.sum()), 'pts': bp[m_pts],
                              'depths': bd[m_pts],
                              'pix': (pv_c[in_b][m_pts], pu_c[in_b][m_pts]),
                              'mean_depth': float(bd[m_pts].mean())
                              if m_pts.any() else 0.0})
                next_local += 1
        labels_by_color[c] = (x0, y0, labels)
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


class _VoxStore:
    """Per-instance sorted voxel sets with lazy compaction and an exact
    per-colour bounding-box index for candidate pruning."""

    def __init__(self):
        self.vox, self.dirty, self.bb, self.col = {}, {}, {}, {}

    @staticmethod
    def bbox(vox):
        idx = np.stack([(vox >> 28) & 0x3FFF, (vox >> 14) & 0x3FFF,
                        vox & 0x3FFF], 1)
        return idx.min(0), idx.max(0)

    def get(self, iid):
        if self.dirty.get(iid):
            self.vox[iid] = np.unique(np.concatenate(
                [self.vox.get(iid, np.zeros(0, np.int64))] + self.dirty[iid]))
            self.dirty[iid] = []
        return self.vox.get(iid, np.zeros(0, np.int64))

    def add(self, iid, c, vox):
        mn, mx = self.bbox(vox)
        cb = self.col.setdefault(c, [[], np.empty((64, 3), np.int64),
                                     np.empty((64, 3), np.int64)])
        if iid in self.bb:
            k = self.bb[iid][2]
            self.bb[iid] = (np.minimum(self.bb[iid][0], mn),
                            np.maximum(self.bb[iid][1], mx), k)
        else:
            k = len(cb[0])
            cb[0].append(iid)
            if k >= len(cb[1]):                      # grow capacity x2
                cb[1] = np.vstack([cb[1], np.empty_like(cb[1])])
                cb[2] = np.vstack([cb[2], np.empty_like(cb[2])])
            self.bb[iid] = (mn, mx, k)
        cb[1][k] = self.bb[iid][0]
        cb[2][k] = self.bb[iid][1]
        self.dirty.setdefault(iid, []).append(vox)
        if sum(len(x) for x in self.dirty[iid]) > 4 * max(
                1, len(self.vox.get(iid, ()))):
            self.get(iid)

    def candidates(self, c, vox):
        """Instances of colour c whose box is within 1 voxel of vox's box
        (overlap counts 6-adjacent voxels only, so nothing else can hit)."""
        cb = self.col.get(c)
        if cb is None:
            return []
        bmn, bmx = self.bbox(vox)
        n = len(cb[0])
        hit = ~((bmn > cb[2][:n] + 1).any(1) | (bmx < cb[1][:n] - 1).any(1))
        return [cb[0][k] for k in np.flatnonzero(hit)]

    def match(self, c, vox):
        best, best_f, merged = None, 0.0, []
        for iid in self.candidates(c, vox):
            f = _overlap_fraction(vox, self.get(iid))
            if f > best_f:
                best, best_f = iid, f
            if f >= THETA_MERGE:
                merged.append(iid)
        return best, best_f, merged

    def finalize(self):
        for iid in list(self.dirty):
            self.get(iid)
        return self.vox


def discover_instances(seq, K, R_wc, t_wc, cfg, log):
    """Pass A. Returns (assign, uf, inst_vox, inst_color, px_per_frame).

    Only *near* observations (depth <= cfg['near_m']) carry identity: a
    blob's near voxels are matched against the instances' near clouds, and
    far voxels of a mixed blob are neither matched nor accumulated (its
    pixels are still labelled with the blob's identity). A blob with no
    near point stays unlabelled (-1 -> background): far views link
    spatially separate copies (a row of chairs seen from 20 m) and any
    label derived from them is ambiguous, so it is not GT."""
    voxel = cfg['voxel']
    near_m = cfg['near_m']
    near = _VoxStore()            # near voxels @ voxel: identity decisions
    inst_color = {}
    assign = {}          # (frame, colour, local_id) -> prov instance id
    px_per_frame = {}    # prov instance id -> {frame: n_px}
    uf = _UF()
    next_id = [0]
    t0 = time.time()

    for i in range(seq.n_frames):
        packed = seq.seg_packed(i)
        depth = seq.depth(i)
        _, _, blobs = extract_blobs(packed, depth, K, R_wc[i], t_wc[i],
                                    cfg['stride'], cfg['split_cell'])
        for b in sorted(blobs, key=lambda b: (-b['n_px'], b['color'],
                                              b['local_id'])):
            c = b['color']
            pts = b['pts']
            if len(pts):
                vox_n = np.unique(_pack_voxels(pts[b['depths'] <= near_m], voxel))
            else:
                vox_n = np.zeros(0, np.int64)
            if len(pts) and not len(vox_n):
                assign[(i, c, b['local_id'])] = -1     # far-only: not GT
                continue
            best, best_f, merged = None, 0.0, []
            anchor = len(vox_n)
            if anchor:
                best, best_f, merged = near.match(c, vox_n)
            if best is not None and best_f >= THETA_ASSIGN:
                iid = best
                for other in merged:
                    uf.union(iid, other)
            elif best is not None and anchor < cfg['min_new_vox']:
                iid = best  # tiny sliver: attach to closest-overlap instance
            elif anchor < cfg['min_new_vox']:
                # sliver with no spatial anchor: leave unlabelled instead of
                # seeding an instance (sliver instances never promote but
                # bloat the per-colour candidate scan)
                assign[(i, c, b['local_id'])] = -1
                continue
            else:
                iid = next_id[0]
                next_id[0] += 1
                inst_color[iid] = c
            if len(vox_n):
                near.add(iid, c, vox_n)
            assign[(i, c, b['local_id'])] = iid
            px_per_frame.setdefault(iid, {})
            px_per_frame[iid][i] = px_per_frame[iid].get(i, 0) + b['n_px']

        if (i + 1) % 200 == 0:
            log(f'  discover {i + 1}/{seq.n_frames} '
                f'instances={len(inst_color)} ({time.time() - t0:.0f}s)')

    inst_vox = near.finalize()
    for iid in inst_color:
        inst_vox.setdefault(iid, np.zeros(0, np.int64))
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
    uf2, n_merged = post_merge(final_vox, final_color, cfg['voxel'],
                               cfg['post_merge_overlap'])
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

    # far labelling: points beyond near_m carry no identity in pass A; in
    # the artifacts they are labelled per point by the nearest promoted
    # object of their colour (within far_label_m of its near cloud), so a
    # row of copies seen from afar is labelled chair by chair and surfaces
    # never seen near stay background.
    from scipy.spatial import cKDTree
    near_m, far_label_m = cfg['near_m'], cfg['far_label_m']
    obj_tree = {}      # colour -> (cKDTree over promoted near clouds, oid per point)
    for c in {final_color[r] for r in root_to_oid}:
        pts_c, oid_c = [], []
        for r, oid in root_to_oid.items():
            if final_color[r] == c and len(final_vox[r]):
                pts_c.append(_unpack_voxels(final_vox[r], cfg['voxel']))
                oid_c.append(np.full(len(final_vox[r]), oid, np.int64))
        if pts_c:
            obj_tree[c] = (cKDTree(np.concatenate(pts_c)), np.concatenate(oid_c))

    def far_oids(c, pts):
        """oid per point (0 = none) by nearest object cloud of colour c."""
        if c not in obj_tree or not len(pts):
            return np.zeros(len(pts), np.int64)
        tree, oid_of_pt = obj_tree[c]
        d, nn = tree.query(pts, k=1, distance_upper_bound=far_label_m)
        out = np.zeros(len(pts), np.int64)
        ok = np.isfinite(d)
        out[ok] = oid_of_pt[nn[ok]]
        return out

    log(f'[{name}] pass B: per-frame artifacts')
    seg_rows = []
    t0 = time.time()
    for i in range(seq.n_frames):
        packed = seq.seg_packed(i)
        depth = seq.depth(i)
        colors, labels_by_color, blobs = extract_blobs(
            packed, depth, K, R_wc[i], t_wc[i], cfg['stride'],
            cfg['split_cell'])

        # observations per final object: list of (ys, xs, pts, depths)
        per_obj = {}
        for b in blobs:
            oid = oid_of(i, b['color'], b['local_id'])
            x0, y0, labels = labels_by_color[b['color']]
            ys, xs = np.nonzero(labels == b['local_id'])
            ys, xs = ys + y0, xs + x0
            pts, dep = b['pts'], b['depths']
            far = dep > near_m
            if not far.any():
                if oid:
                    per_obj.setdefault(oid, []).append((ys, xs, pts, dep))
                continue
            # per-point labels: near points keep the blob identity, far
            # points go to the nearest object (or keep the identity of a
            # mixed blob when nothing is close)
            lab = np.full(len(pts), oid, np.int64)
            fo = far_oids(b['color'], pts[far])
            lab[far] = np.where(fo > 0, fo, oid)
            # pixels follow their nearest strided pixel
            pv, pu = b['pix']
            seeds = np.full(packed.shape, 255, np.uint8)
            seeds[pv, pu] = 0
            _, sl = cv2.distanceTransformWithLabels(
                seeds, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
            seed_oid = np.zeros(sl.max() + 1, np.int64)
            seed_oid[sl[pv, pu]] = lab
            pix_oid = seed_oid[sl[ys, xs]]
            for o in np.unique(lab):
                if o == 0:
                    continue
                mp, ml = pix_oid == o, lab == o
                if mp.any():
                    per_obj.setdefault(int(o), []).append(
                        (ys[mp], xs[mp], pts[ml], dep[ml]))

        sid_of_obj = {oid: sid for sid, oid in
                      enumerate(sorted(per_obj), start=1)}
        if len(sid_of_obj) > 255:
            raise RuntimeError(f'frame {i}: {len(sid_of_obj)} segments > 255')

        seg_img = np.zeros(packed.shape, dtype=np.uint8)
        cloud_pts, offsets, sids, oids = [], [0], [], []
        for oid, obs in sorted(per_obj.items()):
            sid = sid_of_obj[oid]
            for ys, xs, _, _ in obs:
                seg_img[ys, xs] = sid
            pts = np.concatenate([o[2] for o in obs])
            dep = np.concatenate([o[3] for o in obs])
            n_px = int(sum(len(o[0]) for o in obs))
            npts = len(pts)
            seg_rows.append((i, int(seq.stamps_ns[i]), sid, oid, n_px, npts,
                             round(float(dep.mean()), 4) if npts else 0.0,
                             int(is_dynamic_object(oid, seq)),
                             int(npts > 0 and not (dep <= near_m).any())))
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
                    'is_dynamic', 'is_far'])
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
    ap.add_argument('--near', type=float, default=8.0,
                    help='only observations closer than this (m) carry '
                         'identity; blobs with no such point stay unlabelled')
    ap.add_argument('--far-label', type=float, default=0.10,
                    help='points beyond --near are labelled by the nearest '
                         'promoted object cloud within this distance (m)')
    ap.add_argument('--split-cell', type=float, default=0.10,
                    help='3D grid cell (m) for splitting a 2D blob into '
                         'spatially separate sub-blobs (grows with depth)')
    ap.add_argument('--embed-dim', type=int, default=512)
    ap.add_argument('--embed-seed', type=int, default=0)
    a = ap.parse_args()
    cfg = {'stride': a.stride, 'voxel': a.voxel, 'min_px': a.min_px,
           'min_frames': a.min_frames, 'min_new_vox': a.min_new_vox,
           'post_merge_overlap': a.post_merge_overlap,
           'split_cell': a.split_cell, 'near_m': a.near,
           'far_label_m': a.far_label,
           'embed_dim': a.embed_dim, 'embed_seed': a.embed_seed}
    build(a.dataset, a.out, cfg)


if __name__ == '__main__':
    main()
