import os
import sys

import numpy as np
import pandas as pd

import knee_points as K

PX = 1.0 / K.MM
JOINT, KNEECAP, BACK, COR_LOW, COR_HIGH = 0, 1, 2, 3, 4


def series_views(fr, n, joint, kneecap=None):
    row, col, _ = K.to_cache(fr, n, joint)
    if fr["plane"] == 1:
        out = [(COR_LOW, row, col - 30 * PX), (COR_HIGH, row, col + 30 * PX)]
    else:
        if kneecap is not None:
            kr, kc, _ = K.to_cache(fr, n, kneecap)
        elif fr["plane"] == 0:
            kr, kc = row - 25 * PX, col - 40 * PX
        else:
            kr, kc = row - 40 * PX, col
        back = (row + 5 * PX, col + 35 * PX) if fr["plane"] == 0 else (row + 35 * PX, col)
        out = [(JOINT, row, col), (KNEECAP, kr, kc), (BACK,) + back]
    return [(v, float(np.clip(r, 0, K.SIZE - 1)), float(np.clip(c, 0, K.SIZE - 1))) for v, r, c in out]


def build(geo, pts, which):
    pts = pts.set_index("study")
    rows = []
    for r in geo.itertuples():
        fr = K.frame(r)
        p = pts.loc[r.study]
        if which == "vnet":
            joint = np.array([p.vx, p.vy, p.vz])
            kneecap = None if np.isnan(p.px) else np.array([p.px, p.py, p.pz])
        else:
            joint, kneecap = np.array([p.gx, p.gy, p.gz]), None
        n = min(int(r.n_read), K.MAX_SLICES)
        rows += [dict(series=r.series, view=v, row=a, col=b) for v, a, b in series_views(fr, n, joint, kneecap)]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    geo = pd.read_csv(K.GEO)
    pts = pd.read_csv(K.OUT)
    for which in ("vnet", "crossing"):
        df = build(geo, pts, which)
        out = os.path.join(os.path.dirname(K.OUT), f"views_{which}.csv")
        df.to_csv(out, index=False)
        off = np.hypot(df.row - 191.5, df.col - 191.5) * K.MM
        edge = ((df.row < 125) | (df.row > 258) | (df.col < 125) | (df.col > 258)).mean()
        print(which, len(df), "views;", "centre offset mm by view (median):",
              df.assign(off=off).groupby("view").off.median().round(1).to_dict(), f"; boxes reaching past the cache edge: {edge:.1%}")
