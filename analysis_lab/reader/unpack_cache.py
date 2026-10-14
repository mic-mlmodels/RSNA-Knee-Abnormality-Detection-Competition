import glob
import io
import os
import sys
import tarfile

import numpy as np
import pandas as pd


def main():
    src, work = sys.argv[1], sys.argv[2]
    out = f"{work}/cache384_train"
    os.makedirs(out, exist_ok=True)
    tars = sorted(glob.glob(f"{src}/**/cache_s*_*.tar", recursive=True))
    metas = sorted(glob.glob(f"{src}/**/cache_s*_meta.csv", recursive=True))
    print(f"{len(tars)} archives, {len(metas)} meta files", flush=True)
    done = 0
    for t in tars:
        with tarfile.open(t) as tf:
            for m in tf:
                dst = f"{out}/{m.name[:-4]}.npy"
                if not os.path.exists(dst):
                    vol = np.load(io.BytesIO(tf.extractfile(m).read()))["vol"]
                    np.save(dst + ".tmp.npy", vol)
                    os.replace(dst + ".tmp.npy", dst)
                done += 1
        print(f"{os.path.basename(t)} unpacked ({done} series so far)", flush=True)
    meta = pd.concat([pd.read_csv(m) for m in metas]).drop_duplicates("series", keep="last")
    have = {os.path.basename(f)[:-4] for f in glob.glob(f"{out}/*.npy")}
    meta.loc[~meta.series.isin(have), "n"] = 0
    meta.to_csv(f"{work}/cache384_train_meta.csv", index=False)
    errors = int(meta["error"].notna().sum()) if "error" in meta else 0
    print(f"RESULT series_in_meta={len(meta)} files={len(have)} errors={errors} bad_slices={int(meta['n_bad'].fillna(0).sum())}", flush=True)


if __name__ == "__main__":
    main()
