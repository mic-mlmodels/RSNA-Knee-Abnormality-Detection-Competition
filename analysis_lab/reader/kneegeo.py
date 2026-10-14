from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

MM, SIZE, MAX_SLICES = 0.4, 384, 64
CANON = {0: (+1, (1, +1), (2, -1)), 1: (+1, (0, +1), (2, -1)), 2: (-1, (0, +1), (1, +1))}
NET_MM, SPAN_MM, N_SLICES = 0.3125, 4.0, 7
MASS_MIN, PRIOR_MM, AGREE_MM, OBLIQUE_COS, LO, HI = 250, 35.0, 20.0, 0.85, 0.15, 0.85
PX = 1.0 / MM
JOINT, KNEECAP, BACK, COR_LOW, COR_HIGH = 0, 1, 2, 3, 4


def frame(row):
    if isinstance(row, dict):
        row = SimpleNamespace(**row)
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
                fov_row=ps_row * rows, fov_col=ps_col * cols, oblique=float(abs(normal[plane])),
                n=min(int(row.n_read), MAX_SLICES))


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


def vnet_input(vol, fr):
    n = len(vol)
    gap = float(np.linalg.norm(fr["last"] - fr["first"])) / max(1, n - 1)
    step = max(1, int(round(SPAN_MM / max(gap, 1e-3))))
    m = n // 2
    zs = sorted({min(n - 1, max(0, m + j * step)) for j in range(-(N_SLICES // 2), N_SLICES // 2 + 1)})
    mean, std = float(vol.mean()), max(float(vol.std()), 1.0)
    side = int(round(SIZE * MM / NET_MM))
    x = np.stack([cv2.resize((vol[z].astype(np.float32) - mean) / std, (side, side), interpolation=cv2.INTER_LINEAR) for z in zs])
    return zs, x


@torch.no_grad()
def vnet_candidate(net, zs, x, fr, n, cross, dev):
    if fr["oblique"] < OBLIQUE_COS:
        return None
    t = torch.from_numpy(x)[:, None].to(dev)
    h, w = t.shape[-2:]
    t = F.pad(t, (0, (-w) % 16, 0, (-h) % 16), mode="replicate")
    p = net(t).sigmoid()[..., :h, :w]
    p = F.interpolate(p, size=(SIZE, SIZE), mode="bilinear", align_corners=False).cpu().numpy()
    pts, wts, ppts, pwts = [], [], [], []
    for z, pz in zip(zs, p):
        c, m = centroid(pz[1:4].max(0))
        if c is not None:
            pts.append(to_world(fr, z, n, c[0], c[1]))
            wts.append(m)
        cp, mp = centroid(pz[0])
        if cp is not None and mp >= 60:
            ppts.append(to_world(fr, z, n, cp[0], cp[1]))
            pwts.append(mp)
    mass, pmass = int(sum(wts)), int(sum(pwts))
    if mass < MASS_MIN:
        return None
    X = np.average(np.array(pts), axis=0, weights=np.array(wts, float))
    Xp = np.average(np.array(ppts), axis=0, weights=np.array(pwts, float)) if pmass >= 120 else None
    if Xp is not None and not (Xp[1] < X[1] and Xp[2] > X[2] - 10):
        Xp = None
    row, col, _ = to_cache(fr, n, X)
    half = (SIZE - 1) / 2
    rf = 0.5 + (row - half) * MM / fr["fov_row"]
    cf = 0.5 + (col - half) * MM / fr["fov_col"]
    if not (LO <= rf <= HI and LO <= cf <= HI):
        return None
    if np.linalg.norm(X - cross) > PRIOR_MM:
        return None
    return dict(X=X, Xp=Xp, mass=mass, pmass=pmass)


def study_point(cands, cross):
    cs = sorted(cands, key=lambda c: -c["mass"])
    if not cs:
        return cross, None, "crossing"
    close = [c for c in cs if np.linalg.norm(c["X"] - cs[0]["X"]) <= AGREE_MM]
    if len(close) >= 2:
        X = np.average(np.array([c["X"] for c in close]), axis=0, weights=np.array([c["mass"] for c in close], float))
        method = "vnet_avg"
    else:
        X, method = cs[0]["X"], "vnet"
    pc = [c for c in close if c["Xp"] is not None] or [c for c in cs if c["Xp"] is not None]
    Xp = max(pc, key=lambda c: c["pmass"])["Xp"] if pc else None
    return X, Xp, method


def series_views(fr, n, joint, kneecap=None):
    row, col, _ = to_cache(fr, n, joint)
    if fr["plane"] == 1:
        out = [(COR_LOW, row, col - 30 * PX), (COR_HIGH, row, col + 30 * PX)]
    else:
        if kneecap is not None:
            kr, kc, _ = to_cache(fr, n, kneecap)
        elif fr["plane"] == 0:
            kr, kc = row - 25 * PX, col - 40 * PX
        else:
            kr, kc = row - 40 * PX, col
        back = (row + 5 * PX, col + 35 * PX) if fr["plane"] == 0 else (row + 35 * PX, col)
        out = [(JOINT, row, col), (KNEECAP, kr, kc), (BACK,) + back]
    return [(v, float(np.clip(r, 0, SIZE - 1)), float(np.clip(c, 0, SIZE - 1))) for v, r, c in out]
