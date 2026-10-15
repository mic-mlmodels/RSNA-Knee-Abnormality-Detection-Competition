import sys
__file__ = "reader/topk_overlap.py"; exec(open(__file__).read().split("rng = np.random")[0])
import numpy as np
rng = np.random.default_rng(0)
noise = pd.DataFrame(rng.random(sets["ours"].shape) * 1e-6, index=ids, columns=LABELS)
def ov(a, b, k):
    return np.mean([len(set((sets[a][c] + noise[c]).nlargest(k).index) & set((sets[b][c] + noise[c]).nlargest(k).index)) / k for c in LABELS])
pos = np.mean([int((sets["ours"][c] > 0.5).sum()) for c in LABELS])
print("k         " + "  ".join(f"{k:5d}" for k in (25, 50, 100, 250, 500, 1000, 2000)))
for a, b in [("all3", "tsuyu"), ("all3", "dread"), ("tsuyu", "dread"), ("tsuyu", "riadFINAL2")]:
    print(f"{a}-{b:10s}" + "  ".join(f"{ov(a, b, k):.3f}" for k in (25, 50, 100, 250, 500, 1000, 2000)))
print("all3 rows where ours=1 among all3 top 100:", np.mean([(sets["ours"][c].loc[(sets["all3"][c] + noise[c]).nlargest(100).index] > 0.5).mean() for c in LABELS]).round(3))
