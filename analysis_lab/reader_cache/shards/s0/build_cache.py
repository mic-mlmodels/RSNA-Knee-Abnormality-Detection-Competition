import glob
import io
import os
import subprocess
import sys
import tarfile
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

SHARD = 0
N_SHARDS = 5
LIMIT_STUDIES = 0
MAX_OUTPUT_GB = 18.5
TAR_GB = 1.5
OUT = "/kaggle/working"


def find_one(name):
    for depth in range(1, 6):
        hits = glob.glob("/kaggle/input/" + "*/" * depth + name)
        if hits:
            return sorted(hits)[0]
    raise FileNotFoundError(name)


CODE_DIR = os.path.dirname(find_one("preprocess.py"))
target = "/kaggle/working/_pydeps"
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--target", target,
     "--find-links", CODE_DIR, "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"],
    check=False,
)
sys.path.insert(0, target)
sys.path.insert(0, CODE_DIR)
import preprocess as pp

SERIES_CSV = find_one("train_series.csv")
ROOT = os.path.dirname(SERIES_CSV)


def job(args):
    study, series, plane = args
    info = {"study": study, "series": series}
    try:
        vol, extra = pp.load_series(f"{ROOT}/train_series/{study}/{series}", plane)
        info.update(extra)
        buf = io.BytesIO()
        np.savez_compressed(buf, vol=vol)
        info["raw_bytes"] = int(vol.nbytes)
        return info, buf.getvalue()
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"[:200]
        return info, None


def main():
    se = pd.read_csv(SERIES_CSV, dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    studies = sorted(se.StudyInstanceUID.unique())
    mine = [s for i, s in enumerate(studies) if i % N_SHARDS == SHARD]
    if LIMIT_STUDIES:
        mine = mine[:: max(1, len(mine) // LIMIT_STUDIES)][:LIMIT_STUDIES]
    keep = set(mine)
    se = se[se.StudyInstanceUID.isin(keep)]
    tasks = list(zip(se.StudyInstanceUID, se.SeriesInstanceUID, se.Anatomical_Plane))
    print(f"shard {SHARD}/{N_SHARDS}: {len(mine)} of {len(studies)} studies, {len(tasks)} series | size {pp.SIZE} max slices {pp.MAX_SLICES}", flush=True)
    rows, part, tar, tar_bytes, total, t0, stopped = [], 0, None, 0, 0, time.time(), False
    with Pool(4) as pool:
        for i, (info, blob) in enumerate(pool.imap_unordered(job, tasks, chunksize=2)):
            if blob is not None and not stopped:
                if tar is None or tar_bytes > TAR_GB * 1e9:
                    if tar is not None:
                        tar.close()
                    tar = tarfile.open(f"{OUT}/cache_s{SHARD:02d}_{part:03d}.tar", "w")
                    part, tar_bytes = part + 1, 0
                ti = tarfile.TarInfo(info["series"] + ".npz")
                ti.size = len(blob)
                tar.addfile(ti, io.BytesIO(blob))
                tar_bytes += len(blob)
                total += len(blob)
                info["bytes"], info["tar"] = len(blob), part - 1
                if total > MAX_OUTPUT_GB * 1e9:
                    stopped = True
                    print(f"STOPPED WRITING at {total / 1e9:.1f} GB after {i + 1} series", flush=True)
            elif blob is not None:
                info["error"] = "not written: output limit reached"
            rows.append(info)
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(tasks)} series | {total / 1e9:.2f} GB | {time.time() - t0:.0f}s", flush=True)
    if tar is not None:
        tar.close()
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/cache_s{SHARD:02d}_meta.csv", index=False)
    ok = df[df.get("bytes").notna()] if "bytes" in df else df.iloc[:0]
    el = time.time() - t0
    print(f"done: {len(ok)}/{len(df)} series written, errors {int(df['error'].notna().sum()) if 'error' in df else 0}, {total / 1e9:.2f} GB in {el:.0f}s", flush=True)
    if len(ok):
        n_all = 24371
        print(f"per series: {ok['bytes'].mean() / 1e6:.2f} MB compressed, {ok['raw_bytes'].mean() / 1e6:.2f} MB raw, {ok['n'].mean():.1f} slices (median {ok['n'].median():.0f}), bad slices {int(ok['n_bad'].sum())}", flush=True)
        print(f"projection for all {n_all} series: {ok['bytes'].mean() * n_all / 1e9:.0f} GB compressed, {ok['raw_bytes'].mean() * n_all / 1e9:.0f} GB raw, {el / len(df) * n_all / 3600:.1f} h on one 4-core session", flush=True)
        print(df["plane"].value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()
