import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import train as T
from knee import LABELS

GRID = np.round(np.arange(0, 1.01, 0.1), 2)


def auc(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def pct(v):
    return rankdata(v) / len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default="cnxt_ghost_e36")
    ap.add_argument("--full_file", default="oof_e18_fold{f}.csv")
    ap.add_argument("--zoom", default="cnxt_zoom_vnet")
    ap.add_argument("--zoom_file", default="oof_fold{f}.csv")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    folds = [int(v) for v in a.folds.split(",")]
    lab = pd.read_csv(f"{T.WORK}/labels_v0.csv", index_col=0)
    gold_ids = lab[lab.is_gold == 1].index
    gold_y = lab.loc[gold_ids, [f"gold_{c}" for c in LABELS]].values
    rf, rz, ys, fid = [], [], [], []
    gf, gz = [], []
    for f in folds:
        full = pd.read_csv(f"{T.RUNS}/{a.full}/" + a.full_file.format(f=f), index_col=0)
        zoom = pd.read_csv(f"{T.RUNS}/{a.zoom}/" + a.zoom_file.format(f=f), index_col=0)
        ids = full[full.set == "val"].index
        zv = zoom[zoom.set == "val"].loc[ids, LABELS].values
        fv = full.loc[ids, LABELS].values
        rf.append(np.apply_along_axis(pct, 0, fv))
        rz.append(np.apply_along_axis(pct, 0, zv))
        ys.append((lab.loc[ids, LABELS].values > 0.5).astype(int))
        fid.append(np.full(len(ids), f))
        gf.append(1 / (1 + np.exp(-full[full.set == "gold"].loc[gold_ids, LABELS].values)))
        gz.append(1 / (1 + np.exp(-zoom[zoom.set == "gold"].loc[gold_ids, LABELS].values)))
    rf, rz, ys, fid = np.concatenate(rf), np.concatenate(rz), np.concatenate(ys), np.concatenate(fid)
    gf, gz = np.mean(gf, 0), np.mean(gz, 0)

    def best(rows, j):
        return GRID[int(np.argmax([auc(ys[rows, j], (1 - w) * rf[rows, j] + w * rz[rows, j]) for w in GRID]))]

    def best_global(rows):
        return GRID[int(np.argmax([np.nanmean([auc(ys[rows, j], (1 - w) * rf[rows, j] + w * rz[rows, j]) for j in range(12)])
                                   for w in GRID]))]

    def shares(rows):
        g = best_global(rows)
        return np.array([0.5 * best(rows, j) + 0.5 * g for j in range(12)]), g

    nested = np.zeros_like(rf)
    for f in folds:
        inner = fid != f if len(folds) > 1 else fid == f
        w, _ = shares(inner)
        nested[fid == f] = (1 - w) * rf[fid == f] + w * rz[fid == f]
    w, g = shares(np.ones(len(fid), bool))
    print(f"validation studies {len(fid)}, folds {folds}; best single share for all findings: {g}")
    print(f"{'finding':18s} {'full':>7s} {'closeup':>8s} {'nested':>7s} {'share':>6s} | gold: {'full':>6s} {'closeup':>8s} {'blend':>6s}")
    rows = []
    for j, c in enumerate(LABELS):
        gb = (1 - w[j]) * pct(gf[:, j]) + w[j] * pct(gz[:, j])
        r = (auc(ys[:, j], rf[:, j]), auc(ys[:, j], rz[:, j]), auc(ys[:, j], nested[:, j]), w[j],
             auc(gold_y[:, j], gf[:, j]), auc(gold_y[:, j], gz[:, j]), auc(gold_y[:, j], gb))
        rows.append(r)
        print(f"{c:18s} {r[0]:7.4f} {r[1]:8.4f} {r[2]:7.4f} {r[3]:6.2f} |       {r[4]:6.3f} {r[5]:8.3f} {r[6]:6.3f}")
    m = np.nanmean(np.array(rows), 0)
    print(f"{'MEAN':18s} {m[0]:7.4f} {m[1]:8.4f} {m[2]:7.4f} {m[3]:6.2f} |       {m[4]:6.3f} {m[5]:8.3f} {m[6]:6.3f}")
    if a.out:
        json.dump({c: round(float(v), 3) for c, v in zip(LABELS, w)}, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
