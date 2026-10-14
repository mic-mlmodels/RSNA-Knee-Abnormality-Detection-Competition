import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pydicom


def find_one(name):
    for depth in range(1, 6):
        hits = glob.glob("/kaggle/input/" + "*/" * depth + name)
        if hits:
            return sorted(hits)[0]
    raise FileNotFoundError(name)


CSV = find_one("train_series.csv")
ROOT = os.path.dirname(CSV)


def job(args):
    study, series, plane = args
    out = {"series": series, "plane": plane}
    try:
        d = f"{ROOT}/train_series/{study}/{series}"
        files = sorted(os.listdir(d))
        h = pydicom.dcmread(os.path.join(d, files[len(files) // 2]), stop_before_pixels=True)
        ps = [float(v) for v in h.PixelSpacing]
        out.update(n_files=len(files), rows=int(h.Rows), cols=int(h.Columns), ps_row=ps[0], ps_col=ps[1],
                   thickness=float(getattr(h, "SliceThickness", np.nan) or np.nan),
                   acq_matrix=str(getattr(h, "AcquisitionMatrix", "")), field=float(getattr(h, "MagneticFieldStrength", np.nan) or np.nan))
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:120]
    return out


if __name__ == "__main__":
    se = pd.read_csv(CSV, dtype=str)
    tasks = list(zip(se.StudyInstanceUID, se.SeriesInstanceUID, se.Anatomical_Plane))
    with Pool(4) as p:
        rows = p.map(job, tasks, chunksize=50)
    df = pd.DataFrame(rows)
    df.to_csv("/kaggle/working/spacing.csv", index=False)
    ok = df[df.ps_row.notna()]
    print(f"series read: {len(ok)} of {len(df)}")
    ps = ok[["ps_row", "ps_col"]].min(axis=1)
    print("stored pixel size (mm) percentiles 5/25/50/75/95:", np.round(np.percentile(ps, [5, 25, 50, 75, 95]), 3).tolist())
    for lo, hi in ((0, 0.25), (0.25, 0.35), (0.35, 0.45), (0.45, 0.6), (0.6, 9)):
        print(f"  {lo:.2f}-{hi:.2f} mm: {((ps >= lo) & (ps < hi)).mean():.1%}")
    fov = ok.ps_row * ok.rows
    print("field of view (mm) percentiles 5/50/95:", np.round(np.percentile(fov, [5, 50, 95]), 0).tolist())
    print("stored matrix (rows) percentiles 5/50/95:", np.percentile(ok.rows, [5, 50, 95]).tolist())
    print("slice thickness (mm) percentiles 5/50/95:", np.round(np.nanpercentile(ok.thickness, [5, 50, 95]), 1).tolist())
    print("acquisition matrix, most common:", ok.acq_matrix.value_counts().head(6).to_dict())
    print("pixel size by plane (median):", ok.assign(ps=ps).groupby("plane").ps.median().round(3).to_dict())
