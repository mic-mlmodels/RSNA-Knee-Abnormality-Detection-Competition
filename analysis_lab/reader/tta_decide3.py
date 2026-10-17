import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

D = "../data/"
lab = pd.read_csv(D + "reader_work/labels_v0.csv", index_col=0)
P, V, G = {}, [], None
for f in (0, 1):
    z = np.load(D + f"reader_runs/tta_audit3_fold{f}.npz", allow_pickle=True)
    keys, ids = [str(k) for k in z["keys"]], pd.Index(z["ids"].astype(str))
    iv, ig = ids.get_indexer(lab.index[lab.fold == f]), ids.get_indexer(lab.index[lab.is_gold == 1])
    for j, k in enumerate(keys):
        P.setdefault(k, {"val": [], "gold": []})
        P[k]["val"].append(z["preds"][j][iv]); P[k]["gold"].append(z["preds"][j][ig])
    V.append((lab.loc[lab.index[lab.fold == f], LABELS].values > 0.5).astype(int))
    G = lab.loc[lab.index[lab.is_gold == 1], [f"gold_{c}" for c in LABELS]].values.astype(int)
yv = np.concatenate(V)
auc = lambda y, p: float(np.mean([roc_auc_score(y[:, j], p[:, j]) for j in range(12)]))
W = {0: 0.52, 1: 0.48}


def recipe(parts):
    pv = pg = 0
    for o in (0, 1):
        v = g = 0
        for r, fl in parts:
            k = f"plain|k16|off{o}|r{r}|{fl}"
            v, g = v + np.concatenate(P[k]["val"]), g + np.mean(P[k]["gold"], 0)
        pv, pg = pv + W[o] * v / len(parts), pg + W[o] * g / len(parts)
    return pv, pg


R = {"G (8 passes, left-right flip)": recipe([(256, "none"), (256, "hflip"), (320, "none"), (320, "hflip")]),
     "GM (8 passes, mirror)": recipe([(256, "none"), (256, "anat"), (320, "none"), (320, "anat")]),
     "6 passes: 256 plain + 256 mirror + 320 plain": recipe([(256, "none"), (256, "anat"), (320, "none")]),
     "4 passes: 256 plain + 256 mirror": recipe([(256, "none"), (256, "anat")])}
ref_v, ref_g = auc(yv, R["GM (8 passes, mirror)"][0]), auc(G, R["GM (8 passes, mirror)"][1])
rng = np.random.default_rng(0)
boot = [rng.integers(0, len(yv), len(yv)) for _ in range(300)]
for n, (pv, pg) in R.items():
    d = [auc(yv[b], pv[b]) - auc(yv[b], R["GM (8 passes, mirror)"][0][b]) for b in boot]
    print(f"{n:46s} val {auc(yv, pv):.4f} ({auc(yv, pv) - ref_v:+.4f}, 95% {np.percentile(d, 2.5):+.4f}..{np.percentile(d, 97.5):+.4f}) gold {auc(G, pg):.4f} ({auc(G, pg) - ref_g:+.4f})")
ok = lambda n: auc(yv, R[n][0]) >= ref_v - 0.001 and auc(G, R[n][1]) >= ref_g - 0.003
if ok("4 passes: 256 plain + 256 mirror"):
    print("DECISION 4")
elif ok("6 passes: 256 plain + 256 mirror + 320 plain"):
    print("DECISION 6")
else:
    print("DECISION NONE")
