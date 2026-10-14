import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
CACHE = os.path.join(DATA, "reader_work", "cache384_train")
GEO = os.path.join(DATA, "reader_geometry", "geometry.csv")
WEIGHTS = os.path.join(DATA, "reader_geometry", "anatomy.pt")
OUT = os.path.join(DATA, "reader_geometry", "knee_points.csv")

MM, SIZE, MAX_SLICES = 0.4, 384, 64
CANON = {0: (+1, (1, +1), (2, -1)), 1: (+1, (0, +1), (2, -1)), 2: (-1, (0, +1), (1, +1))}
NET_MM, SPAN_MM, N_SLICES = 0.3125, 4.0, 7
MASS_MIN, PRIOR_MM, AGREE_MM, OBLIQUE_COS, LO, HI = 250, 35.0, 20.0, 0.85, 0.15, 0.85


def frame(row):
    iop = np.array([getattr(row, f"iop{i}") for i in range(6)], float)
    r, c = iop[:3], iop[3:]
    normal = np.cross(r, c)
    plane = int(np.abs(normal).argmax())
    order, (cax, csgn), (rax, rsgn) = CANON[plane]
    ps_row, ps_col, rows, cols = row.ps_row, row.ps_col, row.rows, row.cols
    if abs(r[cax]) < abs(r[rax]):
        r, c, ps_row, ps_col, rows, cols = c, r, ps_col, ps_row, cols, rows
    col_dir = r if r[cax] * csgn >= 0 else -r
    row_dir = c if c[rax] * rsgn >= 0 else -c
    lo = np.array([row.lo_x, row.lo_y, row.lo_z])
    hi = np.array([row.hi_x, row.hi_y, row.hi_z])
    first, last = (lo, hi) if order * np.sign(normal[plane]) > 0 else (hi, lo)
    return dict(plane=plane, col_dir=col_dir, row_dir=row_dir, first=first, last=last, normal=normal,
                fov_row=ps_row * rows, fov_col=ps_col * cols, oblique=float(abs(normal[plane])))


def slice_centre(fr, i, n):
    t = 0.5 if n <= 1 else i / (n - 1)
    return fr["first"] + t * (fr["last"] - fr["first"])


def to_world(fr, i, n, row, col):
    half = (SIZE - 1) / 2
    return slice_centre(fr, i, n) + (col - half) * MM * fr["col_dir"] + (row - half) * MM * fr["row_dir"]


def to_cache(fr, n, point):
    half = (SIZE - 1) / 2
    mid = (fr["first"] + fr["last"]) / 2
    d = point - mid
    col = half + float(d @ fr["col_dir"]) / MM
    row = half + float(d @ fr["row_dir"]) / MM
    span = fr["last"] - fr["first"]
    length = float(np.linalg.norm(span))
    z = (n - 1) / 2 if length < 1e-6 or n <= 1 else float((point - fr["first"]) @ span) / length ** 2 * (n - 1)
    return row, col, z


def crossing_point(frames):
    per = {}
    for fr in frames:
        per.setdefault(fr["plane"], []).append((fr["first"] + fr["last"]) / 2)
    g = np.zeros(3)
    for ax, planes in enumerate(((0,), (1,), (0, 1))):
        cs = [c for p in planes for c in per.get(p, [])] or [c for p in per for c in per[p]]
        g[ax] = float(np.median([c[ax] for c in cs]))
    return g


def centroid(prob, thr=0.5):
    m = prob > thr
    if m.sum() < 40:
        return None, 0
    ys, xs = np.nonzero(m)
    w = prob[m]
    return (float((ys * w).sum() / w.sum()), float((xs * w).sum() / w.sum())), int(m.sum())


class Sagittals(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        series, gap = self.items[k]
        vol = np.load(os.path.join(CACHE, series + ".npy"))
        n = len(vol)
        step = max(1, int(round(SPAN_MM / max(gap, 1e-3))))
        m = n // 2
        zs = sorted({min(n - 1, max(0, m + j * step)) for j in range(-(N_SLICES // 2), N_SLICES // 2 + 1)})
        mean, std = float(vol.mean()), max(float(vol.std()), 1.0)
        side = int(round(SIZE * MM / NET_MM))
        x = np.stack([cv2.resize((vol[z].astype(np.float32) - mean) / std, (side, side), interpolation=cv2.INTER_LINEAR) for z in zs])
        return k, n, torch.tensor(zs), torch.from_numpy(x)


def load_net(dev):
    from monai.networks.nets import VNet
    net = VNet(spatial_dims=2, in_channels=1, out_channels=4, act=("elu", {"inplace": True}), dropout_prob_down=0.0,
               dropout_prob_up=(0.0, 0.0), dropout_dim=2, bias=True)
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)["state_dict"]
    state = {k.removeprefix("model.net."): v for k, v in state.items()}
    for name, mod in net.named_modules():
        if name + ".bias" in state and getattr(mod, "bias", 0) is None:
            mod.bias = torch.nn.Parameter(torch.zeros_like(state[name + ".bias"]))
    net.load_state_dict(state, strict=True)
    return net.to(dev).eval().requires_grad_(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dump", default="")
    a = ap.parse_args()
    geo = pd.read_csv(GEO)
    if a.limit:
        geo = geo[geo.study.isin(sorted(geo.study.unique())[:a.limit])]
    frames = {r.series: frame(r) for r in geo.itertuples()}
    study_of = dict(zip(geo.series, geo.study))
    by_study = geo.groupby("study").series.apply(list).to_dict()
    cross = {s: crossing_point([frames[x] for x in ss]) for s, ss in by_study.items()}

    items = []
    for r in geo.itertuples():
        fr = frames[r.series]
        if fr["plane"] == 0 and os.path.exists(os.path.join(CACHE, r.series + ".npy")):
            n = min(int(r.n_read), MAX_SLICES)
            gap = float(np.linalg.norm(fr["last"] - fr["first"])) / max(1, n - 1)
            items.append((r.series, gap))
    dev = "cuda"
    net = load_net(dev)
    loader = torch.utils.data.DataLoader(Sagittals(items), batch_size=None, num_workers=a.workers, prefetch_factor=2)
    cands, why = {}, {}
    for done, (k, n, zs, x) in enumerate(loader):
        series = items[k][0]
        study = study_of[series]
        fr = frames[series]
        if fr["oblique"] < OBLIQUE_COS:
            why.setdefault(study, []).append("oblique")
            continue
        t = x[:, None].to(dev)
        h, w = t.shape[-2:]
        t = F.pad(t, (0, (-w) % 16, 0, (-h) % 16), mode="replicate")
        with torch.autocast("cuda", enabled=False):
            p = net(t).sigmoid()[..., :h, :w]
        p = F.interpolate(p, size=(SIZE, SIZE), mode="bilinear", align_corners=False).cpu().numpy()
        pts, wts, ppts, pwts = [], [], [], []
        for z, pz in zip(zs.tolist(), p):
            c, m = centroid(pz[1:4].max(0))
            if c is not None:
                pts.append(to_world(fr, z, n, c[0], c[1]))
                wts.append(m)
            cp, mp = centroid(pz[0])
            if cp is not None and mp >= 60:
                ppts.append(to_world(fr, z, n, cp[0], cp[1]))
                pwts.append(mp)
        if a.dump and done < 40:
            os.makedirs(a.dump, exist_ok=True)
            np.savez_compressed(os.path.join(a.dump, series + ".npz"), zs=zs.numpy(), p=(p * 255).astype(np.uint8))
        mass, pmass = int(sum(wts)), int(sum(pwts))
        if mass < MASS_MIN:
            why.setdefault(study, []).append(f"mass{mass}")
            continue
        X = np.average(np.array(pts), axis=0, weights=np.array(wts, float))
        Xp = np.average(np.array(ppts), axis=0, weights=np.array(pwts, float)) if pmass >= 120 else None
        if Xp is not None and not (Xp[1] < X[1] and Xp[2] > X[2] - 10):
            Xp = None
        row, col, _ = to_cache(fr, n, X)
        half = (SIZE - 1) / 2
        rf = 0.5 + (row - half) * MM / fr["fov_row"]
        cf = 0.5 + (col - half) * MM / fr["fov_col"]
        if not (LO <= rf <= HI and LO <= cf <= HI):
            why.setdefault(study, []).append("implausible")
            continue
        if np.linalg.norm(X - cross[study]) > PRIOR_MM:
            why.setdefault(study, []).append(f"far{np.linalg.norm(X - cross[study]):.0f}")
            continue
        cands.setdefault(study, []).append(dict(X=X, Xp=Xp, mass=mass, pmass=pmass))
        if done % 1000 == 0:
            print(f"{done} of {len(items)} sagittal series", flush=True)

    rows = []
    for study in by_study:
        g = cross[study]
        cs = sorted(cands.get(study, []), key=lambda c: -c["mass"])
        X, Xp, method = g, None, "crossing"
        if cs:
            close = [c for c in cs if np.linalg.norm(c["X"] - cs[0]["X"]) <= AGREE_MM]
            if len(close) >= 2:
                X = np.average(np.array([c["X"] for c in close]), axis=0, weights=np.array([c["mass"] for c in close], float))
                method = f"vnet_avg{len(close)}"
            else:
                X, method = cs[0]["X"], "vnet"
            pc = [c for c in close if c["Xp"] is not None] or [c for c in cs if c["Xp"] is not None]
            if pc:
                Xp = max(pc, key=lambda c: c["pmass"])["Xp"]
        rows.append(dict(study=study, method=method, n_cands=len(cs), why="|".join(why.get(study, []))[:120],
                         gx=g[0], gy=g[1], gz=g[2], vx=X[0], vy=X[1], vz=X[2],
                         px=np.nan if Xp is None else Xp[0], py=np.nan if Xp is None else Xp[1], pz=np.nan if Xp is None else Xp[2]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT if not a.limit else OUT.replace(".csv", f"_first{a.limit}.csv"), index=False)
    d = np.linalg.norm(df[["vx", "vy", "vz"]].values - df[["gx", "gy", "gz"]].values, axis=1)
    found = df.method != "crossing"
    print(f"studies {len(df)}; V-Net point found for {found.mean():.1%}; kneecap point for {df.px.notna().mean():.1%}")
    print("distance between the two points where the V-Net found one, mm pct 10/50/90/99:",
          np.percentile(d[found], [10, 50, 90, 99]).round(1).tolist())
    print("methods:", df.method.str.replace(r"\d+$", "", regex=True).value_counts().to_dict())


if __name__ == "__main__":
    main()
