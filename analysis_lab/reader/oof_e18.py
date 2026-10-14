import argparse

import numpy as np
import pandas as pd
import torch

import train as T
from knee import LABELS, KneeNet, StudyDataset, build_study_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="cnxt_ghost_e36")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lab = pd.read_csv(f"{T.WORK}/labels_v0.csv", index_col=0)
    meta = pd.read_csv(f"{T.WORK}/cache384_train_meta.csv")
    table = build_study_table(pd.read_csv(f"{T.DATA}/train_series.csv"), dict(zip(meta.series, meta.n.fillna(0))))
    for f in [int(v) for v in a.folds.split(",")]:
        ck = torch.load(f"{T.RUNS}/{a.run}/fold{f}_e18.pt", map_location="cpu", weights_only=False)
        model = KneeNet(ck["args"]["backbone"], pretrained=False)
        model.load_state_dict(ck["model"])
        model.fov = ck["args"].get("fov", 0.92)
        model = model.to(dev).to(memory_format=torch.channels_last).eval()
        rows = []
        for name, part in (("val", lab[(lab.is_gold == 0) & (lab.fold == f)]), ("gold", lab[lab.is_gold == 1])):
            if a.limit:
                part = part.iloc[:a.limit]
            dl = torch.utils.data.DataLoader(StudyDataset(part.index, table, f"{T.WORK}/cache384_train", None, 12),
                                             batch_size=2, num_workers=3)
            out = []
            with torch.no_grad():
                for x, mask, pos, _ in dl:
                    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                        out.append(model(x.to(dev), mask.to(dev), pos.to(dev), res=ck["args"]["res"]).float().cpu())
            df = pd.DataFrame(torch.cat(out).numpy(), index=part.index, columns=LABELS)
            df.insert(0, "set", name)
            rows.append(df)
        out_path = f"{T.RUNS}/{a.run}/oof_e18_fold{f}.csv" if not a.limit else f"{T.RUNS}/oof_e18_test_fold{f}.csv"
        pd.concat(rows).to_csv(out_path)
        print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
