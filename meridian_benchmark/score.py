"""Offline scorer for benchmark runs (no ROS dependency).

Reads a recorder run dir (arrivals.csv, payloads, play_manifest.json) plus
the GT dir and prints/writes the module's metrics from the plan table:

  seg         mask IoU (+ flowtime, drops)
  slam        ATE, RPE (vs gt_poses.csv base)
  clip        flowtime
  geobuilder  flowtime, voxel IoU, outlier ratio
  geotracker/associator/updater/graphcore   TBD (arrival counts only)

Flowtime = recv(output) - max(recv(inputs)) at the same stamp, both taken
by the recorder — an upper bound on module processing time (plan #6).
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import metrics as M

INPUTS = {'seg': ['/camera/rgb'],
          'clip': ['/camera/rgb', '/segment_image'],
          'geobuilder': ['/camera/depth', '/segment_image', '/pose']}
OUTPUTS = {'seg': '/segment_image', 'clip': '/instance_embedding_set',
           'geobuilder': '/instance_3d_set', 'geotracker': '/tracklet_set',
           'associator': '/association_decision_set',
           'updater': '/object_update_set',
           'graphcore': '/local_object_graph_snapshot'}


def load_arrivals(run):
    """topic -> {stamp_ns: recv_mono_ns} (first arrival wins) + counts."""
    first, counts = {}, {}
    with open(Path(run) / 'arrivals.csv') as f:
        for r in csv.DictReader(f):
            t = r['topic']
            counts[t] = counts.get(t, 0) + 1
            first.setdefault(t, {}).setdefault(
                int(r['stamp_ns']), int(r['recv_mono_ns']))
    return first, counts


def load_manifest(run):
    p = Path(run) / 'play_manifest.json'
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def gt_stamp_index(gt):
    """stamp_ns -> frame_idx from GT frames.csv."""
    out = {}
    with open(Path(gt) / 'frames.csv') as f:
        for r in csv.DictReader(f):
            out[int(r['stamp_ns'])] = int(r['frame_idx'])
    return out


def flowtime(arrivals, out_topic, in_topics):
    outs = arrivals.get(out_topic, {})
    dts = []
    for ns, recv in outs.items():
        ins = [arrivals.get(t, {}).get(ns) for t in in_topics]
        if all(v is not None for v in ins):
            dts.append((recv - max(ins)) / 1e9)
    return M.flowtime_stats(dts)


def _played(manifest, arrivals, in_topics):
    if manifest:
        return manifest['n_frames']
    return max((len(arrivals.get(t, {})) for t in in_topics), default=0)


# ---------------- per-module ----------------

def score_seg(gt, run, iou_thresh=0.0):
    import cv2
    arrivals, _ = load_arrivals(run)
    manifest = load_manifest(run)
    idx = gt_stamp_index(gt)
    rec = sorted(Path(run, 'segment_image').glob('*.png'))
    per, missing = [], 0
    for p in rec:
        ns = int(p.stem)
        if ns not in idx:
            missing += 1
            continue
        pred = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        gt_img = cv2.imread(
            str(Path(gt) / 'gt_seg' / f'{idx[ns]:06d}.png'),
            cv2.IMREAD_UNCHANGED)
        per.append(M.frame_mask_iou(pred, gt_img, iou_thresh))
    n_frames = _played(manifest, arrivals, INPUTS['seg'])
    return {
        'mask_iou_mean': float(np.mean([x['mean_matched_iou'] for x in per]))
        if per else 0.0,
        'frames_scored': len(per), 'frames_played': n_frames,
        'frames_dropped': max(0, n_frames - len(per)) + missing,
        'flowtime': flowtime(arrivals, OUTPUTS['seg'], INPUTS['seg'])}


def score_slam(gt, run, tol_s=0.1, rpe_delta_s=1.0):
    pose_csv = Path(run) / 'pose.csv'
    if not pose_csv.exists():
        return {'error': 'no pose.csv recorded', 'n_matched': 0}
    est = np.genfromtxt(pose_csv, delimiter=',', names=True)
    est = np.atleast_1d(est)
    gt_p = np.genfromtxt(Path(gt) / 'gt_poses.csv', delimiter=',',
                         names=True)
    gs = gt_p['stamp_ns'].astype(np.int64)
    es = est['stamp_ns'].astype(np.int64)
    j = np.searchsorted(gs, es)
    j = np.clip(j, 1, len(gs) - 1)
    j = np.where(np.abs(gs[j] - es) < np.abs(gs[j - 1] - es), j, j - 1)
    ok = np.abs(gs[j] - es) <= tol_s * 1e9
    if ok.sum() < 3:
        return {'error': f'only {int(ok.sum())} poses matched GT stamps',
                'n_matched': int(ok.sum())}
    j = j[ok]
    t_est = np.stack([est['tx'], est['ty'], est['tz']], 1)[ok]
    q_est = np.stack([est[k] for k in ('qx', 'qy', 'qz', 'qw')], 1)[ok]
    t_gt = np.stack([gt_p[f'base_t{a}'] for a in 'xyz'], 1)[j]
    q_gt = np.stack([gt_p[f'base_q{a}'] for a in 'xyzw'], 1)[j]
    dt = np.diff(gs[j]).mean() / 1e9 if len(j) > 1 else 1.0
    delta = max(1, int(round(rpe_delta_s / dt)))
    return {'ate': M.ate_rmse(t_est, t_gt),
            'rpe_1frame': M.rpe(t_est, q_est, t_gt, q_gt, delta=1),
            f'rpe_{rpe_delta_s:.0f}s': M.rpe(t_est, q_est, t_gt, q_gt,
                                             delta=delta),
            'n_matched': int(ok.sum()), 'n_est': len(est),
            'coverage': float(ok.sum() / len(gs))}


def score_clip(gt, run):
    arrivals, counts = load_arrivals(run)
    manifest = load_manifest(run)
    n_frames = _played(manifest, arrivals, INPUTS['clip'])
    n_out = counts.get(OUTPUTS['clip'], 0)
    return {'flowtime': flowtime(arrivals, OUTPUTS['clip'], INPUTS['clip']),
            'frames_played': n_frames, 'sets_received': n_out,
            'frames_dropped': max(0, n_frames - n_out)}


def score_geobuilder(gt, run, voxel=0.05, outlier_dist=0.10):
    arrivals, _ = load_arrivals(run)
    manifest = load_manifest(run)
    idx = gt_stamp_index(gt)
    obj_clouds = np.load(Path(gt) / 'gt_object_clouds.npz')
    rec = sorted(Path(run, 'instance_3d_set').glob('*.npz'))
    ious, outliers, seg_hits, seg_gt_total = [], [], 0, 0
    for p in rec:
        ns = int(p.stem)
        if ns not in idx:
            continue
        pred = np.load(p)
        gtf = np.load(Path(gt) / 'gt_frame_clouds' / f'{idx[ns]:06d}.npz')
        gt_of_sid = dict(zip(gtf['segment_ids'].tolist(),
                             range(len(gtf['segment_ids']))))
        oid_of_sid = dict(zip(gtf['segment_ids'].tolist(),
                              gtf['gt_object_ids'].tolist()))
        seg_gt_total += len(gt_of_sid)
        po, go = pred['offsets'], gtf['offsets']
        for k, sid in enumerate(pred['segment_ids'].tolist()):
            if sid not in gt_of_sid:
                continue
            seg_hits += 1
            pp = pred['points'][po[k]:po[k + 1]]
            m = gt_of_sid[sid]
            gp = gtf['points'][go[m]:go[m + 1]]
            ious.append(M.voxel_iou(pp, gp, voxel))
            obj = obj_clouds.get(f'obj_{oid_of_sid[sid]:05d}')
            if obj is not None:
                outliers.append(M.outlier_ratio(pp, obj, outlier_dist))
    n_frames = _played(manifest, arrivals, INPUTS['geobuilder'])
    return {
        'voxel_iou_mean': float(np.mean(ious)) if ious else 0.0,
        'outlier_ratio_mean': float(np.mean(outliers)) if outliers else 0.0,
        'segments_scored': len(ious), 'segment_recall':
        seg_hits / seg_gt_total if seg_gt_total else 0.0,
        'frames_scored': len(rec), 'frames_played': n_frames,
        'frames_dropped': max(0, n_frames - len(rec)),
        'scoring_voxel_m': voxel, 'outlier_dist_m': outlier_dist,
        'flowtime': flowtime(arrivals, OUTPUTS['geobuilder'],
                             INPUTS['geobuilder'])}


def score_tbd(module):
    def score(gt, run):
        _, counts = load_arrivals(run)
        return {'status': 'TBD - arrival counts only',
                'arrivals': counts}
    return score


SCORERS = {'seg': score_seg, 'slam': score_slam, 'clip': score_clip,
           'geobuilder': score_geobuilder,
           'geotracker': score_tbd('geotracker'),
           'associator': score_tbd('associator'),
           'updater': score_tbd('updater'),
           'graphcore': score_tbd('graphcore')}


def score_run(module, gt, run, **kw):
    return SCORERS[module](gt, run, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--module', required=True, choices=sorted(SCORERS))
    ap.add_argument('--gt', required=True)
    ap.add_argument('--run', required=True, help='recorder out dir')
    ap.add_argument('--report', help='write JSON here (default: '
                    '<run>/score.json)')
    a = ap.parse_args()
    result = score_run(a.module, a.gt, a.run)
    report = a.report or str(Path(a.run) / 'score.json')
    with open(report, 'w') as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))
    print(f'-> {report}')


if __name__ == '__main__':
    main()
