"""Verify GT artifacts built by meridian_benchmark.gt_build.

Checks (report -> verify_report.json, non-zero exit on FAIL):
  A structural    file counts, per-frame segment count <= 255, id consistency,
                  embedding sanity (unit norm; same colour -> same vector,
                  different colour -> orthogonal)
  B consistency   sampled frames: gt_seg PNG pixels <-> segments.csv rows
                  <-> raw seg colours <-> re-derived frame-cloud points
  C reprojection  frame-cloud points project back into their own mask
  D cross-frame   compact static object seen from distant frames -> clouds
                  agree in world space (validates pose/intrinsics chain and
                  the instance decomposition together)
  E instances     same-colour promoted instances do not overlap in space
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .uhumans2 import UH2Sequence, unproject


def load_gt(gt_dir):
    gt = {'dir': Path(gt_dir)}
    with open(gt['dir'] / 'config.json') as f:
        gt['config'] = json.load(f)
    with open(gt['dir'] / 'registry.json') as f:
        gt['registry'] = json.load(f)
    poses = np.genfromtxt(gt['dir'] / 'gt_poses.csv', delimiter=',',
                          names=True)
    gt['t_wc'] = np.stack([poses['cam_tx'], poses['cam_ty'],
                           poses['cam_tz']], axis=1)
    gt['q_wc'] = np.stack([poses['cam_qx'], poses['cam_qy'], poses['cam_qz'],
                           poses['cam_qw']], axis=1)
    gt['segments'] = list(csv.DictReader(open(gt['dir'] / 'segments.csv')))
    gt['objects'] = {int(r['gt_object_id']): r for r in
                     csv.DictReader(open(gt['dir'] / 'gt_objects.csv'))}
    return gt


def check_structural(seq, gt, rep):
    n = seq.n_frames
    n_seg_png = len(list((gt['dir'] / 'gt_seg').glob('*.png')))
    n_cloud = len(list((gt['dir'] / 'gt_frame_clouds').glob('*.npz')))
    per_frame = defaultdict(int)
    oids_seen = set()
    for r in gt['segments']:
        per_frame[int(r['frame_idx'])] += 1
        oids_seen.add(int(r['gt_object_id']))
    max_segs = max(per_frame.values()) if per_frame else 0

    emb = np.load(gt['dir'] / 'gt_embeddings.npz')
    E, ids = emb['embeddings'], emb['gt_object_ids']
    norms_ok = np.allclose(np.linalg.norm(E, axis=1), 1.0, atol=1e-4) \
        if len(E) else True
    color_of = {oid: gt['objects'][oid]['color_hex'] for oid in gt['objects']}
    emb_ok, checked = True, 0
    for a in range(0, len(ids), max(1, len(ids) // 40)):
        for b in range(a + 1, len(ids), max(1, len(ids) // 40)):
            same_color = color_of[int(ids[a])] == color_of[int(ids[b])]
            dot = float(E[a] @ E[b])
            if same_color and abs(dot - 1.0) > 1e-4:
                emb_ok = False
            if not same_color and abs(dot) > 1e-4:
                emb_ok = False
            checked += 1

    ok = (n_seg_png == n and n_cloud == n and max_segs <= 255
          and oids_seen <= set(gt['objects']) and norms_ok and emb_ok)
    rep['A_structural'] = {
        'pass': bool(ok), 'frames': n, 'gt_seg_pngs': n_seg_png,
        'frame_clouds': n_cloud, 'max_segments_in_frame': max_segs,
        'undeclared_object_ids': sorted(oids_seen - set(gt['objects']))[:5],
        'embedding_pairs_checked': checked, 'embedding_ok': bool(emb_ok)}
    return ok


def check_consistency(seq, gt, rep, samples):
    """Sampled frames: PNG <-> segments.csv <-> raw colours <-> clouds."""
    K = seq.camera_info('left_cam')['K']
    stride = gt['config']['config']['stride']
    rows_by_frame = defaultdict(dict)
    for r in gt['segments']:
        rows_by_frame[int(r['frame_idx'])][int(r['segment_id'])] = r
    idxs = np.linspace(0, seq.n_frames - 1, samples).astype(int)
    errors = []
    for i in idxs:
        i = int(i)
        seg = cv2.imread(str(gt['dir'] / 'gt_seg' / f'{i:06d}.png'),
                         cv2.IMREAD_UNCHANGED)
        raw = seq.seg_packed(i)
        depth = seq.depth(i)
        rows = rows_by_frame[i]
        sids_png = set(np.unique(seg).tolist()) - {0}
        if sids_png != set(rows):
            errors.append(f'f{i}: sid sets differ png={len(sids_png)} '
                          f'csv={len(rows)}')
            continue
        d = np.load(gt['dir'] / 'gt_frame_clouds' / f'{i:06d}.npz')
        R = Rotation.from_quat(gt['q_wc'][i]).as_matrix()
        for sid, r in rows.items():
            m = seg == sid
            if int(m.sum()) != int(r['n_pixels']):
                errors.append(f'f{i} s{sid}: n_pixels png={int(m.sum())} '
                              f'csv={r["n_pixels"]}')
            cols = np.unique(raw[m])
            want = int(gt['objects'][int(r['gt_object_id'])]['color_hex'], 16)
            if len(cols) != 1 or int(cols[0]) != want:
                errors.append(f'f{i} s{sid}: colour mismatch')
            k = np.where(d['segment_ids'] == sid)[0]
            stored = d['points'][d['offsets'][int(k[0])]:
                                 d['offsets'][int(k[0]) + 1]] if len(k) else \
                np.zeros((0, 3), np.float32)
            pts_cam, _ = unproject(depth, K, stride=stride, valid_mask=m)
            redo = pts_cam @ R.T.astype(np.float32) + \
                gt['t_wc'][i].astype(np.float32)
            if len(redo) != len(stored) or \
                    (len(redo) and float(np.abs(
                        np.sort(redo.ravel()) -
                        np.sort(stored.ravel())).max()) > 1e-3):
                errors.append(f'f{i} s{sid}: cloud re-derivation mismatch '
                              f'({len(stored)} vs {len(redo)} pts)')
    rep['B_consistency'] = {'pass': not errors,
                            'frames_checked': len(idxs),
                            'errors': errors[:10]}
    return not errors


def check_reprojection(seq, gt, rep, samples):
    K = seq.camera_info('left_cam')['K']
    idxs = np.linspace(0, seq.n_frames - 1, samples).astype(int)
    fracs = []
    for i in idxs:
        i = int(i)
        d = np.load(gt['dir'] / 'gt_frame_clouds' / f'{i:06d}.npz')
        if len(d['points']) == 0:
            continue
        seg = cv2.imread(str(gt['dir'] / 'gt_seg' / f'{i:06d}.png'),
                         cv2.IMREAD_UNCHANGED)
        R = Rotation.from_quat(gt['q_wc'][i]).as_matrix()
        p_cam = (d['points'] - gt['t_wc'][i]) @ R
        z = p_cam[:, 2]
        u = np.round(p_cam[:, 0] / z * K[0, 0] + K[0, 2]).astype(int)
        v = np.round(p_cam[:, 1] / z * K[1, 1] + K[1, 2]).astype(int)
        H, W = seg.shape
        inside = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        hit = np.zeros(len(u), bool)
        offs = d['offsets']
        for k, sid in enumerate(d['segment_ids']):
            sl = slice(offs[k], offs[k + 1])
            m = inside[sl]
            if not m.any():
                continue
            mask = cv2.dilate((seg == sid).astype(np.uint8),
                              np.ones((3, 3), np.uint8)) > 0
            h = np.zeros(offs[k + 1] - offs[k], bool)
            h[m] = mask[v[sl][m], u[sl][m]]
            hit[sl] = h
        fracs.append(float(hit.mean()))
    worst = min(fracs) if fracs else 0.0
    rep['C_reprojection'] = {'pass': worst >= 0.995,
                            'frames_checked': len(fracs),
                            'worst_inlier_fraction': worst,
                            'mean_inlier_fraction':
                                float(np.mean(fracs)) if fracs else 0.0}
    return worst >= 0.995


def check_cross_frame(seq, gt, rep, pairs, min_gap=60, min_pts=300,
                      max_diag=2.0):
    compact = set()
    for oid, r in gt['objects'].items():
        d = np.linalg.norm([float(r['aabb_max_x']) - float(r['aabb_min_x']),
                            float(r['aabb_max_y']) - float(r['aabb_min_y']),
                            float(r['aabb_max_z']) - float(r['aabb_min_z'])])
        if d <= max_diag:
            compact.add(oid)
    by_obj = defaultdict(list)
    for r in gt['segments']:
        oid = int(r['gt_object_id'])
        if oid in compact and int(r['n_points_strided']) >= min_pts \
                and not int(r['is_dynamic']):
            by_obj[oid].append((int(r['frame_idx']), int(r['segment_id'])))
    cands = [(oid, obs[0], obs[-1]) for oid, obs in sorted(by_obj.items())
             if obs[-1][0] - obs[0][0] >= min_gap]
    step = max(1, len(cands) // pairs)
    clouds = np.load(gt['dir'] / 'gt_object_clouds.npz')
    near_m = float(gt['config']['config'].get('near_m', np.inf))
    cam = gt['t_wc']
    results = []
    for oid, (fa, sa), (fb, sb) in cands[::step][:pairs]:
        pa = _seg_points(gt, fa, sa)
        pb = _seg_points(gt, fb, sb)
        # identity test: each observation (its near part, the only part
        # that carried identity) must lie on the object's accumulated cloud.
        # Two views of one wide/flat object can be disjoint halves, so the
        # observation-vs-observation distance is reported only as context.
        tree = cKDTree(clouds[f'obj_{oid:05d}'])
        obs_med = []
        for f, p in ((fa, pa), (fb, pb)):
            q = p[np.linalg.norm(p - cam[f], axis=1) <= near_m]
            obs_med.append(float(np.median(tree.query(q, k=1)[0]))
                           if len(q) else float('nan'))
        small, big = (pa, pb) if len(pa) <= len(pb) else (pb, pa)
        dist, _ = cKDTree(big).query(small, k=1)
        results.append({'gt_object_id': oid, 'frame_a': fa, 'frame_b': fb,
                        'gap': fb - fa,
                        'median_nn_m': float(np.nanmax(obs_med)),
                        'obs_to_object_m': obs_med,
                        'pair_median_nn_m': float(np.median(dist)),
                        'p90_nn_m': float(np.percentile(dist, 90))})
    meds = [r['median_nn_m'] for r in results]
    med = float(np.median(meds)) if results else float('nan')
    worst = float(np.max(meds)) if results else float('nan')
    ok = bool(results) and med < 0.05 and worst < 0.10
    rep['D_cross_frame'] = {'pass': ok, 'objects_checked': len(results),
                            'compact_objects': len(compact),
                            'median_of_medians_m': med, 'worst_median_m': worst,
                            'per_object': results}
    return ok


def _seg_points(gt, frame, sid):
    d = np.load(gt['dir'] / 'gt_frame_clouds' / f'{frame:06d}.npz')
    k = int(np.where(d['segment_ids'] == sid)[0][0])
    return d['points'][d['offsets'][k]:d['offsets'][k + 1]]


def check_instance_separation(gt, rep, sample_vox=3000):
    clouds = np.load(gt['dir'] / 'gt_object_clouds.npz')
    by_color = defaultdict(list)
    for oid, r in gt['objects'].items():
        by_color[r['color_hex']].append(oid)
    voxel = gt['config']['config']['voxel']
    worst, n_pairs = 0.0, 0
    offenders = []
    for color, oids in by_color.items():
        if len(oids) < 2:
            continue
        pts = {o: clouds[f'obj_{o:05d}'] for o in oids}
        for a in range(len(oids)):
            for b in range(a + 1, len(oids)):
                pa, pb = pts[oids[a]], pts[oids[b]]
                if not len(pa) or not len(pb):
                    continue
                if len(pa) > len(pb):
                    pa, pb = pb, pa
                if len(pa) > sample_vox:
                    pa = pa[::len(pa) // sample_vox + 1]
                d, _ = cKDTree(pb).query(pa, k=1)
                f = float((d < 1.5 * voxel).mean())
                n_pairs += 1
                if f > worst:
                    worst = f
                if f > 0.2:
                    offenders.append({'color': color,
                                      'oids': [oids[a], oids[b]],
                                      'overlap': f})
    multi = {c: len(o) for c, o in by_color.items() if len(o) > 1}
    rep['E_instances'] = {'pass': not offenders, 'pairs_checked': n_pairs,
                          'worst_overlap_fraction': worst,
                          'colors_with_multiple_instances': len(multi),
                          'max_instances_one_color':
                              max(multi.values()) if multi else 1,
                          'offenders': offenders[:5]}
    return not offenders


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--gt', required=True, help='GT dir for this sequence')
    ap.add_argument('--samples', type=int, default=12)
    ap.add_argument('--pairs', type=int, default=25)
    a = ap.parse_args()

    seq = UH2Sequence(a.dataset)
    gt = load_gt(a.gt)
    rep = {}
    ok = True
    ok &= check_structural(seq, gt, rep)
    ok &= check_consistency(seq, gt, rep, a.samples)
    ok &= check_reprojection(seq, gt, rep, a.samples)
    ok &= check_cross_frame(seq, gt, rep, a.pairs)
    ok &= check_instance_separation(gt, rep)

    with open(Path(a.gt) / 'verify_report.json', 'w') as f:
        json.dump(rep, f, indent=1)

    for k, v in rep.items():
        head = {kk: vv for kk, vv in v.items() if kk != 'per_object'}
        print(('PASS' if v['pass'] else 'FAIL'), k, head)
    print('overall:', 'PASS' if ok else 'FAIL')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
