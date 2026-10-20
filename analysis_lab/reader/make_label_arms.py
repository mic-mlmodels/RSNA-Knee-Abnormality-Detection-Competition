import sys
import numpy as np, pandas as pd
C, W = sys.argv[1], sys.argv[2]
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
lab = pd.read_csv(W + "/labels_v0.csv", index_col=0)
weak = lab.index[lab.is_gold == 0]
ts = pd.read_parquet(C + "/tsuyu122_rsna-knee-mri-balanced-labels-public/labels.parquet").set_index("StudyInstanceUID").loc[weak, LABELS].astype(float)
dr = pd.read_csv(C + "/dreaddevelopment_rsna-knee-labels/labels_llm_soft.csv").set_index("StudyInstanceUID").loc[weak, LABELS].astype(float)
ours = lab.loc[weak, LABELS].astype(float)
def norm(c):
    return "".join(ch for ch in str(c).lower() if ch.isalnum())
want = {norm(c): c for c in LABELS}
want.update({"bakers": "Baker's", "bakerscyst": "Baker's", "pfoa": "PF OA", "patellofemoraloa": "PF OA"})
rj = pd.read_csv(C + "/riadmohamed42_jev-knee-labels/FINAL2_labels_scores.csv")
idc = [c for c in rj.columns if norm(c) in ("studyinstanceuid", "studyuid", "study", "uid", "studyid")][0]
rj = rj.rename(columns={c: want[norm(c)] for c in rj.columns if norm(c) in want}).drop_duplicates(idc).set_index(idc).loc[weak, LABELS].astype(float)
om = pd.read_csv(C + "/../label_candidates2/omerzalman_rsna-knee-11-label-evidence-fusion/calibrated_soft_labels.csv").set_index("StudyInstanceUID").reindex(weak).reindex(columns=LABELS).astype(float)
def mean_skip(*frames):
    return pd.concat(frames).groupby(level=0, sort=False).mean().loc[weak, LABELS]
assert not ts.isna().any().any() and not dr.isna().any().any() and not rj.isna().any().any()
arms = {"tsuyu": ts, "tsuyumix": 0.5 * ours + 0.5 * ts, "oursdread": 0.5 * ours + 0.5 * dr, "dread": dr, "all3": (ours + ts + dr) / 3, "all4": (ours + ts + dr + rj) / 4,
        "all3om": mean_skip(ours, ts, dr, om), "oursts_om": mean_skip(ours, ts, om), "om": om.fillna(ours)}
for f in [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["0", "1"])]:
    keep = lab.loc[weak, "fold"] == f
    for name, t in arms.items():
        out = lab.copy()
        new = t.copy()
        new[keep.values] = ours[keep.values]
        out.loc[weak, LABELS] = new.values
        assert np.allclose(out.loc[weak[keep.values], LABELS].values, ours[keep.values].values)
        assert out.drop(columns=LABELS).equals(lab.drop(columns=LABELS))
        out.to_csv(f"{W}/labels_{name}_f{f}.csv")
        tr = out.loc[weak[~keep.values], LABELS].values
        print(f"labels_{name}_f{f}.csv  train rows {len(tr)}  mean target {tr.mean():.3f}  share above 0.5 {(tr > 0.5).mean():.3f}  val rows kept as ours {int(keep.sum())}")
