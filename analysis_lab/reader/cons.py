import sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
C, W, RUNS = sys.argv[1], sys.argv[2], sys.argv[3]
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
def norm(c):
    return "".join(ch for ch in str(c).lower() if ch.isalnum())
want = {norm(c): c for c in LABELS}
want.update({"bakers": "Baker's", "bakerscyst": "Baker's", "pfoa": "PF OA", "patellofemoraloa": "PF OA"})
def load(path):
    d = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    idc = [c for c in d.columns if norm(c) in ("studyinstanceuid", "studyuid", "study", "uid", "studyid")][0]
    m = {c: want[norm(c)] for c in d.columns if norm(c) in want}
    d = d.rename(columns=m).drop_duplicates(idc).set_index(idc)
    return d[[c for c in LABELS if c in d.columns]].apply(pd.to_numeric, errors="coerce")
lab = pd.read_csv(W + "/labels_v0.csv", index_col=0)
weak = lab[lab.is_gold == 0]
gold = lab[lab.is_gold == 1]
sets = {"ours": weak[LABELS]}
for name, p in [("tsuyu", "tsuyu122_rsna-knee-mri-balanced-labels-public/labels.parquet"),
                ("dread", "dreaddevelopment_rsna-knee-labels/labels_llm_soft.csv"),
                ("codeando", "codeandodev_rsna-knee-etiquetas-debiles/weak_labels.csv"),
                ("gperdrizet", "gperdrizet_rsna-knee-abnormality-pseudolabeled-training-data/pseudolabeled-train.csv"),
                ("yehezkiel", "yehezkielhaganta_psuedo-labeling-knee/data.csv"),
                ("arumpuri", "arumpuri_rsna-knee-weak-labels/weak_labels_v1.csv"),
                ("riadFINAL2", "riadmohamed42_jev-knee-labels/FINAL2_labels_scores.csv"),
                ("karttik", "karttikjangid05_rsna-knee-public-llm-label-sets-compared/consensus_labels.csv")]:
    try:
        d = load(C + "/" + p)
        print(f"{name:11s} cols {d.shape[1]:2d} rows {len(d):5d} weak-covered {len(d.index.intersection(weak.index)):5d} gold-covered {len(d.index.intersection(gold.index)):3d} "
              f"range {np.nanmin(d.values):.2f}..{np.nanmax(d.values):.2f} levels {int(np.median([d[c].nunique() for c in d.columns]))} positives {np.nanmean(d.values > 0.5):.3f}")
        sets[name] = d
    except Exception as e:
        print(name, "FAILED", repr(e)[:200])
print("ours positives", round(float((weak[LABELS].values > 0.5).mean()), 3))
ids = weak.index
for n in list(sets):
    if len(sets[n].index.intersection(ids)) < len(ids) or sets[n].shape[1] < 12:
        print("dropping", n, "(coverage)"); del sets[n]
    else:
        sets[n] = sets[n].loc[ids, LABELS]
names = list(sets)
print("\nmean per-finding correlation between sets")
cm = pd.DataFrame([[np.mean([np.corrcoef(sets[a][c], sets[b][c])[0, 1] for c in LABELS]) for b in names] for a in names], index=names, columns=names)
print(cm.round(2).to_string())
pred = {}
for f in range(5):
    full = pd.read_csv(f"{RUNS}/cnxt_ghost_e36/oof_e18_fold{f}.csv", index_col=0)
    zoom = pd.read_csv(f"{RUNS}/coat_zoom_e18_links/oof_fold{f}.csv", index_col=0)
    v = full[full.set == "val"].index
    r = (np.apply_along_axis(rankdata, 0, full.loc[v, LABELS].values) + np.apply_along_axis(rankdata, 0, zoom[zoom.set == "val"].loc[v, LABELS].values)) / (2 * len(v))
    pred[f] = pd.DataFrame(r, index=v, columns=LABELS)
allp = pd.concat(pred.values())
print("\nAUC of our existing models (trained on OUR labels, out-of-fold, full view + CoAtNet close-up) judged by each label set, 4349 studies")
for n in names:
    y = (sets[n].loc[allp.index].values > 0.5).astype(int)
    a = [roc_auc_score(y[:, j], allp.values[:, j]) if 0 < y[:, j].sum() < len(y) else np.nan for j in range(12)]
    print(f"{n:11s} macro {np.nanmean(a):.4f}  " + " ".join(f"{v:.3f}" for v in a))
