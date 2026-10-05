import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut


IMG = 336
CROP_MM = 140.0
SLOTS = [
    ("Sagittal", 1, 12),
    ("Sagittal", 0, 10),
    ("Coronal", 1, 8),
    ("Coronal", 0, 6),
    ("Axial", -1, 8),
]
MAXS = sum(s[2] for s in SLOTS)
SPANS = [(0.15, 0.85), (0.12, 0.88), (0.10, 0.90), (0.06, 0.94), (0.02, 0.98), (0.0, 1.0)]
N_STUDIES = 24
SEED = 2026
SMALL = 84
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def find_file(name):
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(name)


def order_and_meta(sdir):
    recs, ps_list = [], []
    for f in glob.glob(sdir + "/*.dcm"):
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
            iop = getattr(h, "ImageOrientationPatient", None)
            ipp = getattr(h, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                n = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
                pos = float(np.dot(np.array(ipp, float), n))
            else:
                pos = float(getattr(h, "InstanceNumber", 0) or 0)
            ps = getattr(h, "PixelSpacing", None)
            ps = float(ps[0]) if ps is not None else 0.5
            ps_list.append(ps)
            recs.append((pos, f, ps))
        except Exception:
            recs.append((0.0, f, 0.5))
    recs.sort(key=lambda x: x[0])
    med_ps = float(np.median(ps_list)) if ps_list else 0.5
    return [(f, ps) for _, f, ps in recs], med_ps


def read_px(f):
    d = pydicom.dcmread(f)
    a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def mm_crop_resize(a, ps, size=IMG):
    h, w = a.shape
    cpx = min(int(round(CROP_MM / max(ps, 1e-3))), min(h, w))
    y0, x0 = (h - cpx) // 2, (w - cpx) // 2
    return cv2.resize(a[y0 : y0 + cpx, x0 : x0 + cpx], (size, size), interpolation=cv2.INTER_AREA)


def pick_series(rows, plane, fluid, used):
    cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def build_slot(arrays, files, med_ps, k, span):
    n = len(files)
    lo, hi = int(n * span[0]), int(n * span[1]) - 1
    hi = max(hi, lo)
    picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else np.zeros(k, int)
    picks = np.minimum(picks, n - 1)
    valid = [arrays[p] for p in picks if arrays[p] is not None]
    if valid:
        loq, hiq = np.percentile(np.concatenate([a.ravel() for a in valid]), [2.0, 98.0])
    else:
        loq, hiq = 0.0, 1.0
    out = np.zeros((k, IMG, IMG), np.uint8)
    for j, p in enumerate(picks):
        a = arrays[p]
        if a is None:
            continue
        ps = files[p][1]
        aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
        out[j] = (mm_crop_resize(aw, ps if ps > 0 else med_ps) * 255).astype(np.uint8)
    return out, picks


def corr(a, b):
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a, b = a - a.mean(), b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else 0.0


def main():
    vols = np.load(find_file("all_vols.npy"), mmap_mode="r")
    masks = np.load(find_file("all_masks.npy"))
    ids = np.load(find_file("all_ids.npy"), allow_pickle=True).astype(str)
    series_csv = find_file("train_series.csv")
    root = os.path.dirname(series_csv)
    tser = pd.read_csv(series_csv, dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    ser = {k: v.to_dict("records") for k, v in tser.groupby("StudyInstanceUID", sort=False)}
    log(f"corpus {vols.shape} | series table {len(tser)} rows | data root {root}")

    rng = np.random.default_rng(SEED)
    rows_idx = sorted(rng.choice(len(ids), size=N_STUDIES, replace=False).tolist())
    pool = ThreadPoolExecutor(max_workers=8)
    agg = {s: {"exact": [], "mad": [], "corr": []} for s in SPANS}
    presence = {"both": 0, "corpus_only": 0, "builder_only": 0, "neither": 0}
    inferred, examples, per_study = [], [], []

    for count, row in enumerate(rows_idx):
        sid = ids[row]
        theirs = np.asarray(vols[row])
        tmask = masks[row]
        used, offset = set(), 0
        study = {"study": sid, "row": int(row), "slots": []}
        for plane, fluid, k in SLOTS:
            block = theirs[offset : offset + k]
            has_corpus = bool(tmask[offset : offset + k].sum() > 0)
            rec = pick_series(ser.get(sid, []), plane, fluid, used)
            files, med_ps = ([], 0.5)
            if rec is not None:
                used.add(rec["SeriesInstanceUID"])
                files, med_ps = order_and_meta(f"{root}/train_series/{sid}/{rec['SeriesInstanceUID']}")
            has_builder = bool(files)
            key = ("both" if has_builder else "corpus_only") if has_corpus else ("builder_only" if has_builder else "neither")
            presence[key] += 1
            slot = {"plane": plane, "fluid": fluid, "k": k, "presence": key, "n_files": len(files)}
            if key == "both":
                def safe(f):
                    try:
                        return read_px(f)
                    except Exception:
                        return None
                arrays = list(pool.map(safe, [f for f, _ in files]))
                for span in SPANS:
                    mine, picks = build_slot(arrays, files, med_ps, k, span)
                    d = np.abs(block.astype(np.int16) - mine.astype(np.int16))
                    cs = [corr(block[j], mine[j]) for j in range(k)]
                    agg[span]["exact"].append(float((d == 0).mean()))
                    agg[span]["mad"].append(float(d.mean()))
                    agg[span]["corr"].append(float(np.mean(cs)))
                    slot[f"span_{span[0]}_{span[1]}"] = {
                        "exact": float((d == 0).mean()),
                        "mad": float(d.mean()),
                        "corr": float(np.mean(cs)),
                        "picks": picks.tolist(),
                    }
                small = []
                for a, (_, ps) in zip(arrays, files):
                    if a is None:
                        small.append(None)
                        continue
                    lo_v, hi_v = np.percentile(a, [2.0, 98.0])
                    aw = np.clip((a - lo_v) / (hi_v - lo_v + 1e-6), 0, 1)
                    small.append(mm_crop_resize(aw, ps if ps > 0 else med_ps, SMALL))
                matched, best = [], []
                for j in range(k):
                    ref = cv2.resize(block[j], (SMALL, SMALL), interpolation=cv2.INTER_AREA)
                    scores = [corr(ref, s) if s is not None else -1.0 for s in small]
                    matched.append(int(np.argmax(scores)))
                    best.append(float(np.max(scores)))
                n = len(files)
                slot["matched_indices"] = matched
                slot["matched_corr"] = [round(b, 3) for b in best]
                if np.mean(best) > 0.9 and n > 1:
                    inferred.append({"n": n, "k": k, "lo": min(matched) / n, "hi": (max(matched) + 1) / n, "monotonic": bool(np.all(np.diff(matched) >= 0))})
                if len(examples) < 10:
                    examples.append({"plane": plane, "n": n, "k": k, "matched": matched, "corr": round(float(np.mean(best)), 3)})
            study["slots"].append(slot)
            offset += k
        per_study.append(study)
        log(f"study {count + 1}/{N_STUDIES} done")

    print("\n=== slot presence (corpus mask vs builder) ===")
    print(presence)
    print("\n=== candidate slice ranges, averaged over slots present in both ===")
    print(f"{'range':>12} {'exact match':>12} {'mean abs diff':>14} {'mean corr':>10}")
    summary = {}
    for span in SPANS:
        a = agg[span]
        summary[f"{span[0]}-{span[1]}"] = {m: float(np.mean(a[m])) for m in a}
        print(f"{span[0]:>5.2f}-{span[1]:<5.2f} {np.mean(a['exact']):>12.4f} {np.mean(a['mad']):>14.3f} {np.mean(a['corr']):>10.4f}")
    print("\n=== which DICOM slice each corpus slice matches best (first 10 slots) ===")
    for e in examples:
        print(e)
    if inferred:
        lo = np.array([i["lo"] for i in inferred])
        hi = np.array([i["hi"] for i in inferred])
        print(f"\n=== inferred range from {len(inferred)} confidently matched slots ===")
        print(f"first picked slice / n: median {np.median(lo):.3f} (10th-90th pct {np.percentile(lo, 10):.3f}-{np.percentile(lo, 90):.3f})")
        print(f"(last picked slice + 1) / n: median {np.median(hi):.3f} (10th-90th pct {np.percentile(hi, 10):.3f}-{np.percentile(hi, 90):.3f})")
        print(f"matched order increasing in {np.mean([i['monotonic'] for i in inferred]):.0%} of slots")
    json.dump({"presence": presence, "summary": summary, "inferred": inferred, "studies": per_study}, open("/kaggle/working/parity.json", "w"))
    log("done")


if __name__ == "__main__":
    main()
