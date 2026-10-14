import sys

import numpy as np
import pandas as pd

import train as T
from knee import LABELS

folds = [int(v) for v in sys.argv[2].split(",")]
k = pd.read_csv(sys.argv[1], index_col=0)
lab = pd.read_csv(f"{T.WORK}/labels_v0.csv", index_col=0)
ids = [u for u in k.index if lab.loc[u, "is_gold"] == 1]
local = []
for f in folds:
    d = pd.read_csv(f"{T.RUNS}/cnxt_zoom_vnet/oof_fold{f}.csv", index_col=0)
    local.append(d[d.set == "gold"].loc[ids, LABELS].values)
local = np.mean(local, 0)
kag = k.loc[ids, LABELS].values
corr = [float(np.corrcoef(local[:, j], kag[:, j])[0, 1]) for j in range(12)]
diff = float(np.abs(local - kag).mean())
print(f"parity on {len(ids)} gold studies, folds {folds}: logit correlation min {min(corr):.4f} mean {np.mean(corr):.4f}; mean abs logit difference {diff:.4f}")
ok = min(corr) > 0.98 and diff < 0.25
print("PARITY", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
