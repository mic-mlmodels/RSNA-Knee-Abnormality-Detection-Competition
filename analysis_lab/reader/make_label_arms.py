import sys
import numpy as np, pandas as pd
C, W = sys.argv[1], sys.argv[2]
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
lab = pd.read_csv(W + "/labels_v0.csv", index_col=0)
weak = lab.index[lab.is_gold == 0]
ts = pd.read_parquet(C + "/tsuyu122_rsna-knee-mri-balanced-labels-public/labels.parquet").set_index("StudyInstanceUID").loc[weak, LABELS].astype(float)
dr = pd.read_csv(C + "/dreaddevelopment_rsna-knee-labels/labels_llm_soft.csv").set_index("StudyInstanceUID").loc[weak, LABELS].astype(float)
ours = lab.loc[weak, LABELS].astype(float)
assert not ts.isna().any().any() and not dr.isna().any().any()
arms = {"tsuyu": ts, "tsuyumix": 0.5 * ours + 0.5 * ts, "oursdread": 0.5 * ours + 0.5 * dr, "dread": dr, "all3": (ours + ts + dr) / 3}
for f in (0, 1):
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
