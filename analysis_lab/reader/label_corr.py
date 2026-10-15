import sys
import numpy as np, pandas as pd
__file__ = "reader/topk_overlap.py"
exec(open(__file__).read().split("rng = np.random")[0])
sets["oursdread"] = (sets["ours"] + sets["dread"]) / 2
for n in ["tsuyu", "tsuyumix", "all3", "oursdread", "dread", "riadFINAL2", "karttik", "codeando", "yehezkiel", "arumpuri"]:
    r = np.mean([sets["ours"][c].corr(sets[n][c], method="spearman") for c in LABELS])
    p = np.mean([sets["ours"][c].corr(sets[n][c]) for c in LABELS])
    print(f"{n:11s} pearson {p:.3f} spearman {r:.3f}")
