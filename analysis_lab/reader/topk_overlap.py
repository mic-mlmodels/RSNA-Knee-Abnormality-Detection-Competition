import sys
import numpy as np, pandas as pd
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2], ""]
src = open(__file__.replace("topk_overlap.py", "cons.py")).read().split("names = list(sets)")[0]
exec(src)
sets["all3"] = (sets["ours"] + sets["tsuyu"] + sets["dread"]) / 3
sets["tsuyumix"] = (sets["ours"] + sets["tsuyu"]) / 2
names = ["ours", "all3", "tsuyumix", "tsuyu", "dread", "riadFINAL2", "karttik", "codeando", "yehezkiel", "arumpuri"]
names = [n for n in names if n in sets]
rng = np.random.default_rng(0)
noise = pd.DataFrame(rng.random(sets["ours"].shape) * 1e-6, index=ids, columns=LABELS)
K = {c: int((sets["ours"][c] > 0.5).sum()) for c in LABELS}
top = {n: {c: set((sets[n][c] + noise[c]).nlargest(K[c]).index) for c in LABELS} for n in names}
ov = pd.DataFrame([[np.mean([len(top[a][c] & top[b][c]) / K[c] for c in LABELS]) for b in names] for a in names], index=names, columns=names)
print(f"studies {len(ids)}; k per finding = our positive count: " + ", ".join(f"{c} {K[c]}" for c in LABELS))
print(f"chance overlap (random top-k) averaged over findings: {np.mean([K[c] / len(ids) for c in LABELS]):.3f}")
print("\nshare of the top-k studies shared between two label sets, mean over the 12 findings")
print(ov.round(3).to_string())
print("\nper finding, against ours")
print(pd.DataFrame({n: [len(top['ours'][c] & top[n][c]) / K[c] for c in LABELS] for n in names[1:]}, index=LABELS).round(2).to_string())
