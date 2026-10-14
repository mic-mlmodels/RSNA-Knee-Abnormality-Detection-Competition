import argparse
import os

import numpy as np
import pandas as pd
import torch

import kneegeo as G
from knee import LABELS, N_SLOTS, N_VIEWS, KneeNet, build_study_table, window_centres
from preprocess import MAX_SLICES, SIZE, load_series, series_geometry

PASSES = 3


class Studies(torch.utils.data.Dataset):
    def __init__(self, studies, table, series_of, root, k):
        self.studies, self.table, self.series_of, self.root, self.k = list(studies), table, series_of, root, k

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        st = self.studies[i]
        slots = self.table.get(st, [[] for _ in range(N_SLOTS)])
        x = np.zeros((N_SLOTS, self.k, 3, SIZE, SIZE), np.uint8)
        mask, pos = np.zeros(N_SLOTS, bool), np.zeros((N_SLOTS, self.k), np.float32)
        chosen = {cands[0]: s for s, cands in enumerate(slots) if cands}
        frames, slot_frame, sag = {}, {}, []
        for sr, plane in self.series_of.get(st, []):
            sdir = f"{self.root}/{st}/{sr}"
            try:
                if sr in chosen or plane == "Sagittal":
                    vol, info = load_series(sdir)
                    geo = info.get("geometry")
                else:
                    vol, geo = None, series_geometry(sdir)
            except Exception:
                continue
            fr = G.frame(geo) if geo is not None else None
            if fr is not None:
                frames[sr] = fr
            if vol is not None and fr is not None and fr["plane"] == 0 and len(vol) == fr["n"]:
                sag.append((fr,) + G.vnet_input(vol, fr))
            if sr in chosen and vol is not None and len(vol) >= 3:
                s, n = chosen[sr], len(vol)
                c = window_centres(n, self.k, False)
                x[s] = np.stack([vol[np.clip([j - 1, j, j + 1], 0, n - 1)] for j in c])
                mask[s], pos[s] = True, c / max(n - 1, 1)
                slot_frame[s] = (sr, fr, n)
        return dict(study=st, x=torch.from_numpy(x), mask=torch.from_numpy(mask), pos=torch.from_numpy(pos),
                    frames=frames, slot_frame=slot_frame, sag=sag)


def place_views(item, net, dev):
    ctr = np.zeros((PASSES, N_SLOTS, 2), np.float32)
    vid = np.zeros((PASSES, N_SLOTS), np.int64)
    rows, method = [], "none"
    if item["frames"]:
        cross = G.crossing_point(list(item["frames"].values()))
        cands = []
        for fr, zs, xin in item["sag"]:
            c = G.vnet_candidate(net, zs, xin, fr, fr["n"], cross, dev)
            if c is not None:
                cands.append(c)
        joint, kneecap, method = G.study_point(cands, cross)
        for s, (sr, fr, n) in item["slot_frame"].items():
            if fr is None:
                continue
            v = G.series_views(fr, n, joint, kneecap)
            for p in range(PASSES):
                view, r, c = v[p % len(v)]
                vid[p, s] = view
                ctr[p, s] = ((r - 191.5) / 192.0, (c - 191.5) / 192.0)
            rows += [dict(study=item["study"], series=sr, view=a, row=b, col=c, method=method) for a, b, c in v]
    return torch.from_numpy(ctr), torch.from_numpy(vid), rows


def load_models(ckpts, dev):
    models = []
    for path in ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = KneeNet(ck["args"]["backbone"].split(".")[0], pretrained=False, n_views=N_VIEWS)
        m.load_state_dict(ck["model"])
        m.fov = ck["args"]["fov"]
        m.mean, m.std = ck.get("norm", (m.mean, m.std))
        models.append((m.to(dev).to(memory_format=torch.channels_last).eval(), ck["args"]["res"]))
    return models


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("split")
    ap.add_argument("out_csv")
    ap.add_argument("ckpts", nargs="*")
    ap.add_argument("--vnet", required=True)
    ap.add_argument("--views_out", default="")
    ap.add_argument("--studies", default="")
    ap.add_argument("--logits", action="store_true")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    se = pd.read_csv(f"{a.data_dir}/{a.split}_series.csv")
    root = f"{a.data_dir}/{a.split}_series"
    if a.studies:
        studies = [s.strip() for s in open(a.studies) if s.strip()]
    else:
        studies = pd.read_csv(f"{a.data_dir}/{a.split}.csv").StudyInstanceUID.tolist()
    se = se[se.StudyInstanceUID.isin(set(studies))]
    n_files, series_of = {}, {}
    for st, sr, pl in zip(se.StudyInstanceUID, se.SeriesInstanceUID, se.Anatomical_Plane):
        d = f"{root}/{st}/{sr}"
        n_files[sr] = min(len(os.listdir(d)), MAX_SLICES) if os.path.isdir(d) else 0
        if n_files[sr]:
            series_of.setdefault(st, []).append((sr, pl))
    table = build_study_table(se, n_files)
    net = torch.jit.load(a.vnet, map_location=dev).eval()
    models = load_models(a.ckpts, dev)
    dl = torch.utils.data.DataLoader(Studies(studies, table, series_of, root, a.k), batch_size=a.bs, num_workers=a.workers,
                                     collate_fn=lambda b: b)
    out, view_rows, methods = [], [], {}
    for batch in dl:
        placed = [place_views(item, net, dev) for item in batch]
        for item, pl in zip(batch, placed):
            view_rows += pl[2]
            m = pl[2][0]["method"] if pl[2] else "none"
            methods[m] = methods.get(m, 0) + 1
        if not models:
            continue
        x = torch.stack([b["x"] for b in batch]).to(dev)
        mask = torch.stack([b["mask"] for b in batch]).to(dev)
        pos = torch.stack([b["pos"] for b in batch]).to(dev)
        ctr = torch.stack([p[0] for p in placed]).to(dev)
        vid = torch.stack([p[1] for p in placed]).to(dev)
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            logit = torch.stack([torch.stack([m(x, mask, pos, res=res, centre=ctr[:, p], vid=vid[:, p]).float()
                                              for p in range(PASSES)]).mean(0) for m, res in models])
        out.append((logit.mean(0) if a.logits else logit.sigmoid().mean(0)).cpu())
    print("knee point methods:", methods, flush=True)
    if a.views_out:
        pd.DataFrame(view_rows).to_csv(a.views_out, index=False)
    if models:
        pd.DataFrame(torch.cat(out).numpy(), index=pd.Index(studies, name="StudyInstanceUID"), columns=LABELS).to_csv(a.out_csv)
        print("wrote", a.out_csv)


if __name__ == "__main__":
    main()
