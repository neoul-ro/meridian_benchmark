"""Message-agnostic metric primitives for the benchmark scorer.

All functions take plain numpy arrays; message/file parsing lives in the
scorer. Convention: clouds are float32 [N,3] world-frame metres.
"""

import numpy as np


# ---------------- masks (seg) ----------------

def label_iou_matrix(pred, gt):
    """IoU matrix between all pred labels and gt labels (label 0 = ignore).

    pred/gt: same-shape integer label images.
    Returns (pred_ids, gt_ids, iou[len(pred_ids), len(gt_ids)]).
    """
    pred = np.asarray(pred).ravel()
    gt = np.asarray(gt).ravel()
    pred_ids = np.unique(pred)
    pred_ids = pred_ids[pred_ids != 0]
    gt_ids = np.unique(gt)
    gt_ids = gt_ids[gt_ids != 0]
    if not len(pred_ids) or not len(gt_ids):
        return pred_ids, gt_ids, np.zeros((len(pred_ids), len(gt_ids)))
    pi = np.searchsorted(pred_ids, pred)
    gi = np.searchsorted(gt_ids, gt)
    valid = np.isin(pred, pred_ids) & np.isin(gt, gt_ids)
    inter = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.int64)
    np.add.at(inter, (pi[valid], gi[valid]), 1)
    area_p = np.bincount(pi[np.isin(pred, pred_ids)],
                         minlength=len(pred_ids)).astype(np.int64)
    area_g = np.bincount(gi[np.isin(gt, gt_ids)],
                         minlength=len(gt_ids)).astype(np.int64)
    union = area_p[:, None] + area_g[None, :] - inter
    return pred_ids, gt_ids, inter / np.maximum(union, 1)


def greedy_match_iou(iou, thresh=0.0):
    """Greedy 1-1 matching on an IoU matrix, highest IoU first.

    Returns list of (pred_idx, gt_idx, iou) with iou > thresh.
    """
    iou = iou.copy()
    matches = []
    while iou.size:
        k = np.argmax(iou)
        i, j = np.unravel_index(k, iou.shape)
        if iou[i, j] <= thresh:
            break
        matches.append((int(i), int(j), float(iou[i, j])))
        iou[i, :] = -1
        iou[:, j] = -1
    return matches


def frame_mask_iou(pred, gt, thresh=0.0):
    """Mean matched IoU + counts for one frame of label images.

    Returns dict: mean_matched_iou (over GT segments; unmatched GT count as
    0), n_gt, n_pred, n_matched.
    """
    _, gt_ids, iou = label_iou_matrix(pred, gt)
    matches = greedy_match_iou(iou, thresh)
    total = sum(m[2] for m in matches)
    n_gt = len(gt_ids)
    return {'mean_matched_iou': total / n_gt if n_gt else 1.0,
            'n_gt': int(n_gt), 'n_pred': int(iou.shape[0]),
            'n_matched': len(matches)}


# ---------------- clouds (geobuilder) ----------------

def voxelize(pts, voxel):
    """Point cloud -> set of packed voxel indices (int64)."""
    if len(pts) == 0:
        return np.zeros(0, np.int64)
    idx = np.floor(np.asarray(pts, np.float64) / voxel).astype(np.int64)
    HALF = 1 << 20
    np.clip(idx, -HALF, HALF - 1, out=idx)
    return np.unique((idx[:, 0] + HALF) * (1 << 42)
                     + (idx[:, 1] + HALF) * (1 << 21) + (idx[:, 2] + HALF))


def voxel_iou(pts_a, pts_b, voxel):
    """IoU of the voxelized occupancy of two clouds."""
    va = voxelize(pts_a, voxel)
    vb = voxelize(pts_b, voxel)
    if not len(va) and not len(vb):
        return 1.0
    inter = len(np.intersect1d(va, vb, assume_unique=True))
    return inter / (len(va) + len(vb) - inter)


def outlier_ratio(pts, gt_pts, max_dist):
    """Fraction of pts farther than max_dist from any GT point."""
    if len(pts) == 0:
        return 0.0
    if len(gt_pts) == 0:
        return 1.0
    from scipy.spatial import cKDTree
    d, _ = cKDTree(np.asarray(gt_pts)).query(np.asarray(pts), k=1)
    return float((d > max_dist).mean())


# ---------------- timing ----------------

def flowtime_stats(dt_seconds):
    """Summary stats for per-frame flowtime samples (seconds)."""
    d = np.asarray(dt_seconds, dtype=np.float64)
    if not len(d):
        return {'n': 0}
    return {'n': int(len(d)), 'mean': float(d.mean()),
            'median': float(np.median(d)),
            'p90': float(np.percentile(d, 90)),
            'p99': float(np.percentile(d, 99)), 'max': float(d.max())}
