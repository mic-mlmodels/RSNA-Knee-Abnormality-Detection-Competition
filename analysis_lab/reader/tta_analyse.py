import itertools
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from knee import LABELS

D = "../data/"
lab = pd.read_csv(D + "reader_work/labels_v0.csv", index_col=0)
folds = [int(f) for f in sys.argv[1:]] or [0, 1]
P, V, G = {}, [], None
for f in folds:
    z = np.load(D + f"reader_runs/tta_audit_fold{f}.npz", allow_pickle=True)
    keys, ids = [str(k) for k in z["keys"]], pd.Index(z["ids"].astype(str))
    val = lab.index[(lab.fold == f)]
    gold = lab.index[lab.is_gold == 1]
    iv, ig = ids.get_indexer(val), ids.get_indexer(gold)
    for j, k in enumerate(keys):
        P.setdefault(k, {"val": [], "gold": []})
        P[k]["val"].append(z["preds"][j][iv])
        P[k]["gold"].append(z["preds"][j][ig])
    V.append((lab.loc[val, LABELS].values > 0.5).astype(int))
    G = lab.loc[gold, [f"gold_{c}" for c in LABELS]].values.astype(int)
yv = np.concatenate(V)


def get(key):
    return np.concatenate(P[key]["val"]), np.mean(P[key]["gold"], 0)


def auc(y, p):
    return float(np.mean([roc_auc_score(y[:, j], p[:, j]) for j in range(12)]))


def combo(table="plain", k=16, offs=(0,), ress=(256,), flips=("none",), wo=None):
    pv, pg = 0, 0
    wo = wo or {o: 1 / len(offs) for o in offs}
    for o in offs:
        v = g = 0
        for r, fl in itertools.product(ress, flips):
            a, b = get(f"{table}|k{k}|off{o}|r{r}|{fl}")
            v, g = v + a, g + b
        n = len(ress) * len(flips)
        pv, pg = pv + wo[o] * v / n, pg + wo[o] * g / n
    return pv, pg


base_v, base_g = combo(k=12)
rng = np.random.default_rng(0)
boot = [rng.integers(0, len(yv), len(yv)) for _ in range(300)]


def row(name, pv, pg):
    d = [auc(yv[b], pv[b]) - auc(yv[b], base_v[b]) for b in boot]
    print(f"{name:58s} val {auc(yv, pv):.4f} ({auc(yv, pv) - auc(yv, base_v):+.4f}, 95% {np.percentile(d, 2.5):+.4f} to {np.percentile(d, 97.5):+.4f}) | gold {auc(G, pg):.4f} ({auc(G, pg) - auc(G, base_g):+.4f})")


print(f"folds {folds}: {len(yv)} validation studies, {len(G)} gold studies (gold = average of the folds' models)")
row("baseline: 12 windows, 256 px, no extras (what scored 0.946)", base_v, base_g)
print("-- one change at a time")
row("16 windows per series", *combo())
row("16 windows + fill missing series slots", *combo(table="fill"))
row("16 windows + second stride", *combo(offs=(0, 1)))
row("16 windows + 320 px instead of 256", *combo(ress=(320,)))
row("16 windows + 256 and 320 px averaged", *combo(ress=(256, 320)))
row("16 windows + left-right flip of every series", *combo(flips=("none", "hflip")))
row("16 windows + anatomical mirror (sagittal reversed)", *combo(flips=("none", "anat")))
print("-- combinations")
full = combo(table="fill", offs=(0, 1), ress=(256, 320), flips=("none", "hflip"), wo={0: 0.52, 1: 0.48})
row("the notebook's full recipe (all five together)", *full)
row("full recipe without the slot fill", *combo(offs=(0, 1), ress=(256, 320), flips=("none", "hflip"), wo={0: 0.52, 1: 0.48}))
row("full recipe without the flip", *combo(table="fill", offs=(0, 1), ress=(256, 320)))
row("full recipe without 320 px", *combo(table="fill", offs=(0, 1), flips=("none", "hflip")))
row("full recipe without the second stride", *combo(table="fill", ress=(256, 320), flips=("none", "hflip")))
row("16 windows + both strides + both resolutions, no fill/flip", *combo(offs=(0, 1), ress=(256, 320)))
print("-- per finding, the notebook's full recipe minus baseline (validation | gold)")
for j, c in enumerate(LABELS):
    dv = roc_auc_score(yv[:, j], full[0][:, j]) - roc_auc_score(yv[:, j], base_v[:, j])
    dg = roc_auc_score(G[:, j], full[1][:, j]) - roc_auc_score(G[:, j], base_g[:, j])
    print(f"   {c:18s} {dv:+.4f} | {dg:+.4f}")
