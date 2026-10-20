import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import knee
from knee import LABELS, N_VIEWS, KneeNet, StudyDataset, build_study_table, mirror_augment

D = "../data/"
WORK = D + "reader_work"
CACHE = WORK + "/cache384_train"
VIEWS = D + "reader_geometry/views_vnet.csv"
_centres = knee.window_centres


class OffsetStudies(StudyDataset):
    def __init__(self, *a, offset=0, **kw):
        super().__init__(*a, **kw)
        self.offset = offset

    def __getitem__(self, i):
        if self.offset:
            knee.window_centres = lambda n, k, train: np.clip(_centres(n, k, train) + self.offset, 1, max(1, n - 2))
        try:
            return super().__getitem__(i)
        finally:
            knee.window_centres = _centres


def tta_variants(model, b0, b1, res):
    def run(b, mirror):
        x, mask, pos, ctr, vid = b
        out = []
        for p in range(ctr.shape[1]):
            xx, pp, c, v = mirror_augment(x, pos, 1.0, ctr[:, p], vid[:, p]) if mirror else (x, pos, ctr[:, p], vid[:, p])
            out.append(model(xx, mask, pp, res=res, centre=c, vid=v).float())
        return torch.stack(out).mean(0)

    o0, m0, o1, m1 = run(b0, False), run(b0, True), run(b1, False), run(b1, True)
    return {"base": o0, "mirror_only": m0, "offset1_only": o1,
            "mirror": (o0 + m0) / 2, "offset": 0.52 * o0 + 0.48 * o1,
            "offset_mirror": 0.52 * (o0 + m0) / 2 + 0.48 * (o1 + m1) / 2}


@torch.no_grad()
def main():
    fold, run = int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "cnxt_zoom_all3_mirror24"
    ck = torch.load(f"{D}reader_runs/{run}/fold{fold}.pt", map_location="cpu", weights_only=False)
    a = ck["args"]
    model = KneeNet(a["backbone"], pretrained=False, n_views=N_VIEWS, spatial=a.get("spatial", False))
    model.load_state_dict(ck["model"])
    model.fov = a["fov"]
    model = model.cuda().to(memory_format=torch.channels_last).eval()
    lab = pd.read_csv(f"{WORK}/labels_v0.csv", index_col=0)
    meta = pd.read_csv(f"{WORK}/cache384_train_meta.csv")
    table = build_study_table(pd.read_csv(D + "competition/train_series.csv"), dict(zip(meta.series, meta.n.fillna(0))))
    vt = pd.read_csv(VIEWS)
    vt["r"], vt["c"] = (vt.row - 191.5) / 192.0, (vt.col - 191.5) / 192.0
    views = {sr: d[["view", "r", "c"]].values.astype(np.float32) for sr, d in vt.groupby("series")}
    studies = list(lab[((lab.fold == fold) & (lab.is_gold == 0)) | (lab.is_gold == 1)].index)
    dls = [torch.utils.data.DataLoader(OffsetStudies(studies, table, CACHE, None, 12, views=views, offset=o),
                                       batch_size=4, num_workers=4, prefetch_factor=2) for o in (0, 1)]
    outs, t0 = {}, time.time()
    for i, (r0, r1) in enumerate(zip(*dls)):
        b0 = [t.cuda(non_blocking=True) for j, t in enumerate(r0) if j != 3]
        b1 = [t.cuda(non_blocking=True) for j, t in enumerate(r1) if j != 3]
        with torch.autocast("cuda", dtype=torch.float16):
            v = tta_variants(model, b0, b1, a["res"])
        assert "base" in v, "tta_variants must return a 'base' entry"
        for k, p in v.items():
            outs.setdefault(k, []).append(p.float().cpu())
        if i % 50 == 0:
            print(f"fold {fold}: batch {i} of {len(dls[0])} ({time.time() - t0:.0f}s)", flush=True)
    keys = list(outs)
    preds = np.stack([torch.cat(outs[k]).numpy() for k in keys]).astype(np.float32)
    np.savez_compressed(f"{D}reader_runs/tta_audit_zoom_{run}_fold{fold}.npz", keys=np.array(keys), preds=preds, ids=np.array(studies))
    ids = pd.Index(studies)
    iv, ig = ids.get_indexer(lab.index[(lab.fold == fold) & (lab.is_gold == 0)]), ids.get_indexer(lab.index[lab.is_gold == 1])
    yv = (lab.iloc[lab.index.get_indexer(ids[iv])][LABELS].values > 0.5).astype(int)
    yg = lab.iloc[lab.index.get_indexer(ids[ig])][[f"gold_{c}" for c in LABELS]].values.astype(int)
    auc = lambda y, p: float(np.mean([roc_auc_score(y[:, j], p[:, j]) for j in range(12)]))
    base = preds[keys.index("base")]
    rng = np.random.default_rng(0)
    boot = [rng.integers(0, len(iv), len(iv)) for _ in range(300)]
    print(f"fold {fold}: {len(iv)} val, {len(ig)} gold studies, {time.time() - t0:.0f}s")
    for k, p in zip(keys, preds):
        d = [auc(yv[b], p[iv][b]) - auc(yv[b], base[iv][b]) for b in boot]
        print(f"{k:30s} val {auc(yv, p[iv]):.4f} ({auc(yv, p[iv]) - auc(yv, base[iv]):+.4f}, 95% {np.percentile(d, 2.5):+.4f}..{np.percentile(d, 97.5):+.4f})"
              f"  gold {auc(yg, p[ig]):.4f} ({auc(yg, p[ig]) - auc(yg, base[ig]):+.4f})")


if __name__ == "__main__":
    main()
