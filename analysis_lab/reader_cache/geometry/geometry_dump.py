import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pydicom

TAGS = ["ImageOrientationPatient", "ImagePositionPatient", "PixelSpacing", "Rows", "Columns", "NumberOfFrames",
        "SliceThickness", "SeriesDescription"]


def find_one(name):
    for depth in range(1, 6):
        hits = glob.glob("/kaggle/input/" + "*/" * depth + name)
        if hits:
            return sorted(hits)[0]
    raise FileNotFoundError(name)


CSV = find_one("train_series.csv")
ROOT = os.path.dirname(CSV)


def header(path):
    try:
        h = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=TAGS)
        return ([float(v) for v in h.ImageOrientationPatient], [float(v) for v in h.ImagePositionPatient],
                [float(v) for v in h.PixelSpacing], int(h.Rows), int(h.Columns), int(getattr(h, "NumberOfFrames", 1) or 1),
                float(getattr(h, "SliceThickness", np.nan) or np.nan), str(getattr(h, "SeriesDescription", ""))[:60])
    except Exception:
        return None


def job(args):
    study, series, plane = args
    out = {"study": study, "series": series, "plane_csv": plane}
    try:
        d = f"{ROOT}/train_series/{study}/{series}"
        files = sorted(os.listdir(d))
        with ThreadPoolExecutor(8) as ex:
            hs = [h for h in ex.map(header, [os.path.join(d, f) for f in files]) if h is not None]
        out.update(n_files=len(files), n_read=len(hs))
        if not hs:
            return out
        iop = np.array(hs[0][0])
        r, c = iop[:3], iop[3:]
        normal = np.cross(r, c)
        ps, rows, cols = hs[0][2], hs[0][3], hs[0][4]
        centres = np.array([np.array(h[1]) + r * ps[1] * (cols - 1) / 2 + c * ps[0] * (rows - 1) / 2 for h in hs])
        proj = centres @ normal
        lo, hi = centres[proj.argmin()], centres[proj.argmax()]
        same = all(np.allclose(h[0], hs[0][0], atol=1e-3) and h[3] == rows and h[4] == cols for h in hs)
        out.update(rows=rows, cols=cols, ps_row=ps[0], ps_col=ps[1], frames=max(h[5] for h in hs), thickness=hs[0][6],
                   desc=hs[0][7], same_geometry=int(same), gap=float(np.median(np.abs(np.diff(np.sort(proj))))) if len(proj) > 1 else 0.0,
                   **{f"iop{i}": float(v) for i, v in enumerate(iop)},
                   **{f"lo_{a}": float(v) for a, v in zip("xyz", lo)}, **{f"hi_{a}": float(v) for a, v in zip("xyz", hi)})
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:120]
    return out


if __name__ == "__main__":
    t0 = time.time()
    se = pd.read_csv(CSV, dtype=str)
    tasks = list(zip(se.StudyInstanceUID, se.SeriesInstanceUID, se.Anatomical_Plane))
    rows = []
    with Pool(4) as p:
        for i, r in enumerate(p.imap(job, tasks, chunksize=20)):
            rows.append(r)
            if i % 2000 == 0:
                print(f"{i} of {len(tasks)} series, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("/kaggle/working/geometry.csv", index=False)
    ok = df[df.rows.notna()]
    print(f"series with geometry: {len(ok)} of {len(df)}; multi-frame: {(ok.frames > 1).sum()}; mixed geometry: {(ok.same_geometry == 0).sum()}")
    print(f"done in {time.time() - t0:.0f}s")
