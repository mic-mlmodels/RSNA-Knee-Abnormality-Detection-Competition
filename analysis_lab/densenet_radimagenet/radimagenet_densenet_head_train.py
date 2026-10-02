import re
import h5py
import torch.nn.functional as F
import hashlib
from concurrent.futures import ThreadPoolExecutor
import gc, os, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn


T0 = time.time()


SEED = 2026


TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


GROUP = 3


HDR_THREADS = 16


PIX_THREADS = 12


ORDER_THREADS = 32


ORDER_BUDGET_S = 5400


LAT_MIN_OFFSET_MM = 20.0


RULES_LEGACY = {
    "order": "dominant_axis",
    "lat": "corner_x",
    "slot_fallback": True,
    "decode_fill": "zero",
}


LEGACY_LAT_OFFSET_MM = 5.0


FATSAT_OPTS = {"FS", "FATSAT", "FAT_SAT", "FSAT"}


_SEP = re.compile("[_\\-.]")


_FATSAT_RX = re.compile(
    "\\bfs\\b|fatsat|fat sat|\\bstir\\b|\\bspair\\b|\\bspir\\b|\\bwe\\b|water excit|\\btirm\\b|\\bsting\\b|\\bfatsup\\b"
)


_T1_RX = re.compile("\\bt1\\b|\\bt1w\\b")


_T2_RX = re.compile("\\bt2\\b|\\bt2w\\b")


_PD_RX = re.compile("\\bpd\\b|\\bpdw\\b|proton|\\bdp\\b|dens")


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def find_root():
    for c in [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
        Path("data"),
        Path("."),
    ]:
        if (c / "test.csv").is_file() and (c / "test_series").is_dir():
            return c
    base = Path("/kaggle/input")
    if base.is_dir():
        for depth1 in sorted((p for p in base.iterdir() if p.is_dir())):
            for cand in [depth1] + sorted((p for p in depth1.iterdir() if p.is_dir())):
                if (cand / "test.csv").is_file():
                    return cand
    raise FileNotFoundError(
        f"competition mount not found (cwd {Path.cwd()}); expected a directory holding test.csv and test_series/"
    )


ROOT = find_root()


HDR_TAGS = [
    "SeriesDescription",
    "SequenceName",
    "ScanOptions",
    "ScanningSequence",
    "RepetitionTime",
    "EchoTime",
    "Laterality",
    "PixelSpacing",
    "Rows",
    "Columns",
    "RescaleSlope",
    "RescaleIntercept",
    "ImagePositionPatient",
    "ImageOrientationPatient",
]


def _hdr_vec(s, n):
    if not isinstance(s, str):
        return None
    try:
        v = [float(x) for x in s.split("|")]
    except ValueError:
        return None
    return np.array(v) if len(v) >= n else None


def side_from_geometry(h):
    cx = {}
    for r in h.itertuples(index=False):
        ipp = _hdr_vec(getattr(r, "ImagePositionPatient", None), 3)
        iop = _hdr_vec(getattr(r, "ImageOrientationPatient", None), 6)
        ps = _hdr_vec(getattr(r, "PixelSpacing", None), 2)
        rows, cols = (getattr(r, "Rows", None), getattr(r, "Columns", None))
        if ipp is None or iop is None or ps is None or (not rows) or (not cols):
            continue
        try:
            c = (
                ipp[:3]
                + iop[:3] * ps[1] * float(cols) / 2
                + iop[3:6] * ps[0] * float(rows) / 2
            )
        except (TypeError, ValueError):
            continue
        cx.setdefault(r.StudyInstanceUID, []).append(float(c[0]))
    out = {}
    for st, xs in cx.items():
        m = float(np.median(xs))
        out[st] = None if abs(m) < LAT_MIN_OFFSET_MM else "R" if m < 0 else "L"
    return out


def side_from_corner_x(h):
    out = {}
    for st, g in h.groupby("StudyInstanceUID"):
        xs = []
        for r in g.itertuples(index=False):
            ipp = _hdr_vec(getattr(r, "ImagePositionPatient", None), 3)
            if ipp is not None and np.isfinite(ipp).all():
                xs.append(float(ipp[0]))
        if not xs:
            out[st] = None
            continue
        x = float(np.median(xs))
        out[st] = None if abs(x) < LEGACY_LAT_OFFSET_MM else "R" if x < 0 else "L"
    return out


def lat_of(h, tag=""):
    geo = side_from_corner_x(h) if RULES["lat"] == "corner_x" else side_from_geometry(h)
    d, n_tag, n_geo, n_none, n_disagree = ({}, 0, 0, 0, 0)
    for st, g in h.groupby("StudyInstanceUID"):
        v = [str(x).strip().upper() for x in g["Laterality"].dropna()]
        if RULES["lat"] == "corner_x" and "ImageLaterality" in g.columns:
            v += [str(x).strip().upper() for x in g["ImageLaterality"].dropna()]
        v = [x[0] for x in v if x and x[0] in ("L", "R")]
        side = v[0] if v else None
        if side is not None:
            n_tag += 1
            if geo.get(st) is not None and geo[st] != side:
                n_disagree += 1
        else:
            side = geo.get(st)
            n_geo += side is not None
            n_none += side is None
        d[st] = side
    log(
        f"{tag}laterality: {n_tag} from the tag, {n_geo} from geometry, {n_none} unresolved; tag and geometry disagree on {n_disagree} ({n_disagree / max(n_tag, 1):.1%} of the tagged)"
    )
    return d


def probe(item):
    split, study, series, path = item
    row = {
        "split": split,
        "StudyInstanceUID": study,
        "SeriesInstanceUID": series,
        "dir": path,
    }
    try:
        files = sorted((e.name for e in os.scandir(path) if e.name.endswith(".dcm")))
        row["files"] = files
        row["n_slices"] = len(files)
        if not files:
            return row
        ds = pydicom.dcmread(
            os.path.join(path, files[len(files) // 2]),
            stop_before_pixels=True,
            force=True,
        )
        for t in HDR_TAGS:
            v = getattr(ds, t, None)
            if v is None:
                row[t] = None
            elif isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
                row[t] = "|".join((str(x) for x in v))
            else:
                row[t] = str(v)
    except Exception as exc:
        row["err"] = str(exc)[:120]
    return row


def walk(split):
    base = ROOT / split
    items = []
    if not base.is_dir():
        return pd.DataFrame(
            columns=[
                "split",
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "dir",
                "files",
                "n_slices",
            ]
            + HDR_TAGS
        )
    for study in os.scandir(base):
        if study.is_dir():
            for series in os.scandir(study.path):
                if series.is_dir():
                    items.append((split, study.name, series.name, series.path))
    with ThreadPoolExecutor(max_workers=HDR_THREADS) as pool:
        rows = list(pool.map(probe, items))
    return pd.DataFrame(rows)


def annotate(df):
    desc = df["SeriesDescription"].fillna("") + " " + df["SequenceName"].fillna("")
    desc = desc.str.lower().str.replace(_SEP, " ", regex=True)
    opts = df["ScanOptions"].fillna("").str.upper().str.split("|")
    opts_fs = opts.apply(lambda ts: any((t.strip() in FATSAT_OPTS for t in ts)))
    df["fatsat"] = desc.str.contains(_FATSAT_RX) | opts_fs
    tr = pd.to_numeric(df["RepetitionTime"], errors="coerce")
    te = pd.to_numeric(df["EchoTime"], errors="coerce")
    gre = df["ScanningSequence"].fillna("").str.upper().str.contains("GR")
    t1, t2, pdw = (
        desc.str.contains(_T1_RX),
        desc.str.contains(_T2_RX),
        desc.str.contains(_PD_RX),
    )
    df["weight"] = np.where(
        t1 & ~t2 & ~pdw,
        "T1",
        np.where(
            t2 & ~pdw,
            "T2",
            np.where(
                pdw,
                "PD",
                np.where(
                    gre,
                    "GRE",
                    np.where(
                        tr < 800,  # type: ignore
                        "T1",
                        np.where(te > 60, "T2", np.where(tr >= 800, "PD", "UNK")),  # type: ignore
                    ),
                ),
            ),
        ),
    )
    df["fluid"] = np.isin(df["weight"], ["PD", "T2"])
    df["px"] = pd.to_numeric(
        df["PixelSpacing"].fillna("").str.split("|").str[0].replace("", np.nan),
        errors="coerce",
    )
    return df


def pick_slots(series_df, plane_map):
    series_df = series_df.copy()
    series_df["plane"] = series_df["SeriesInstanceUID"].map(plane_map)
    out = {}
    for study, g in series_df.groupby("StudyInstanceUID"):
        chosen = {}
        for name, plane, fluid, fs in SLOTS:
            sel = (g["plane"] == plane) & (g["fatsat"] == fs)
            if fluid is not None:
                sel &= g["fluid"] == fluid
            cand = g[sel]
            if len(cand) == 0 and RULES["slot_fallback"] and (fluid is False):
                cand = g[(g["plane"] == plane) & ~g["fatsat"]]
            if len(cand):
                chosen[name] = cand.sort_values("n_slices", ascending=False).iloc[0]
        out[study] = chosen
    return out


ORDER_TAGS = [(32, 50), (32, 55), (32, 19)]


DECODE_FAILED = []


def _natural_key(name):
    return tuple(
        (int(x) if x.isdigit() else x.lower() for x in re.split("(\\d+)", str(name)))
    )


def _order_dominant_axis(rec):
    files, d = (rec["files"], rec["dir"])
    rows = []
    for pos, f in enumerate(files):
        ipp = inst = None
        try:
            ds = pydicom.dcmread(
                os.path.join(d, f),
                force=True,
                stop_before_pixels=True,
                specific_tags=["ImagePositionPatient", "InstanceNumber"],
            )
            raw = getattr(ds, "ImagePositionPatient", None)
            if raw is not None and len(raw) >= 3:
                c = np.asarray(raw[:3], dtype=np.float64)
                if np.isfinite(c).all():
                    ipp = c
            n = getattr(ds, "InstanceNumber", None)
            if n is not None:
                inst = float(n)
        except Exception:
            pass
        rows.append((f, ipp, inst, pos))
    placed = [r for r in rows if r[1] is not None]
    need = max(2, int(0.8 * len(rows)))
    if len(placed) >= need:
        xyz = np.stack([r[1] for r in placed])
        axis = int(np.argmax(np.ptp(xyz, axis=0)))
        spare = float(np.nanmedian(xyz[:, axis]))
        rows.sort(
            key=lambda r: (
                float(r[1][axis]) if r[1] is not None else spare,
                r[2] if r[2] is not None else float("inf"),
                r[3],
            )
        )
    elif sum((r[2] is not None for r in rows)) >= need:
        rows.sort(key=lambda r: (r[2] if r[2] is not None else float("inf"), r[3]))
    else:
        rows.sort(key=lambda r: _natural_key(r[0]))
    return ([r[0] for r in rows], True)


def order_slices(rec):
    if RULES["order"] == "dominant_axis":
        return _order_dominant_axis(rec)
    files, d = (rec["files"], rec["dir"])
    keyed = []
    for f in files:
        k = None
        try:
            ds = pydicom.dcmread(
                os.path.join(d, f),
                force=True,
                stop_before_pixels=True,
                specific_tags=ORDER_TAGS,
            )
            iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
            ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
            k = float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
        except Exception:
            try:
                k = float(ds.InstanceNumber)  # type: ignore
            except Exception:
                k = None
        keyed.append((k, f))
    if any((k is None for k, _ in keyed)):
        return (files, False)
    return ([f for _, f in sorted(keyed, key=lambda t: t[0])], True)


def read_slot(rec, n_slice=None, out_size=None):
    n_slice = GROUP if n_slice is None else n_slice
    out_size = IMG if out_size is None else out_size
    files, d, px = (rec.get("ordered") or rec["files"], rec["dir"], rec["px"])
    n = len(files)
    if n == 0:
        return None
    lo, hi = (int(SLICE_BAND[0] * (n - 1)), int(SLICE_BAND[1] * (n - 1)))
    idx = (
        np.unique(np.linspace(lo, hi, n_slice).astype(int))
        if hi > lo
        else np.array([n // 2])
    )
    while len(idx) < n_slice:
        idx = np.append(idx, idx[-1])
    planes = []
    for i in idx[:n_slice]:
        try:
            ds = pydicom.dcmread(os.path.join(d, files[int(i)]), force=True)
            a = ds.pixel_array.astype(np.float32)
            sl = float(getattr(ds, "RescaleSlope", 1) or 1)
            ic = float(getattr(ds, "RescaleIntercept", 0) or 0)
            a = a * sl + ic
        except Exception:
            a = None
        planes.append(a)
    got = [k for k, p in enumerate(planes) if p is not None]
    if RULES["decode_fill"] == "zero":
        if not got:
            DECODE_FAILED.append(rec.get("SeriesInstanceUID", d))
        planes = [
            np.zeros((out_size, out_size), np.float32) if p is None else p
            for p in planes
        ]
        got = list(range(len(planes)))
    if not got:
        DECODE_FAILED.append(rec.get("SeriesInstanceUID", d))
        return None
    if len(got) < len(planes):
        DECODE_FAILED.append(rec.get("SeriesInstanceUID", d))
        for k, p in enumerate(planes):
            if p is None:
                planes[k] = planes[min(got, key=lambda j: abs(j - k))]
    shp = planes[0].shape
    planes = [p if p.shape == shp else np.zeros(shp, np.float32) for p in planes]
    vol = np.stack(planes)
    if px and np.isfinite(px) and (px > 0):
        want = int(round(CROP_MM / px))
        h, w = shp
        if 16 < want < min(h, w):
            cy, cx = (h // 2, w // 2)
            half = want // 2
            vol = vol[:, max(0, cy - half) : cy + half, max(0, cx - half) : cx + half]
    lo_v, hi_v = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo_v) / max(hi_v - lo_v, 1e-06), 0, 1)
    t = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)
    t = F.interpolate(
        t, size=(out_size, out_size), mode="bilinear", align_corners=False
    )
    return (t.squeeze(0) * 255).round().clamp(0, 255).to(torch.uint8)


def normalise_laterality(img, plane, lat):
    if lat != "R":
        return img
    if plane in ("Coronal", "Axial"):
        return torch.flip(img, dims=[-1])
    return torch.flip(img, dims=[0])


ORDER_CACHE = os.environ.get("RSNA_ORDER_CACHE") or None


def build_cache(slot_map, plane_map, lat_map, tag):
    studies = sorted(slot_map)
    sidx = {s: i for i, s in enumerate(studies)}
    cache = np.zeros((len(studies), N_SLOT, CACHE_SLICES, IMG, IMG), np.uint8)
    mask = np.zeros((len(studies), N_SLOT), np.float32)
    log(f"{tag}: cache {cache.shape} = {cache.nbytes / 1024**3:.1f} GB")
    jobs = [
        (st, k, plane, slot_map[st][name])
        for st in studies
        for k, (name, plane, _, _) in enumerate(SLOTS)
        if name in slot_map[st]
    ]
    n_job = len(jobs)
    t_ord = time.time()
    n_slice_total = sum((len(j[3]["files"]) for j in jobs))
    log(f"{tag}: ordering {len(jobs)} slot-series ({n_slice_total} slice headers)")
    ok = done = 0
    CHUNK_O = 1024
    seen = {}
    if ORDER_CACHE and Path(ORDER_CACHE).is_file():
        try:
            import json as _json

            seen = _json.loads(Path(ORDER_CACHE).read_text())
        except (OSError, ValueError):
            seen = {}
        hit = 0
        for _, _, _, rec in jobs:
            e = seen.get(rec["SeriesInstanceUID"])
            if e and len(e["files"]) == len(rec["files"]):
                rec["ordered"] = e["files"]
                ok += int(e["good"])
                hit += 1
        jobs = [j for j in jobs if "ordered" not in j[3]]
        log(f"{tag}: {hit} slot-series ordered from {ORDER_CACHE}, {len(jobs)} to read")
    with ThreadPoolExecutor(max_workers=ORDER_THREADS) as pool:
        for c0 in range(0, len(jobs), CHUNK_O):
            block = jobs[c0 : c0 + CHUNK_O]
            for (_, _, _, rec), (files, good) in zip(
                block, pool.map(lambda j: order_slices(j[3]), block)
            ):
                rec["ordered"] = files
                ok += int(good)
                done += 1
                if ORDER_CACHE:
                    seen[rec["SeriesInstanceUID"]] = {
                        "files": files,
                        "good": bool(good),
                    }
            budget = min(
                ORDER_BUDGET_S, max(60.0, (TIME_BUDGET - (time.time() - T0)) * 0.35)
            )
            if time.time() - t_ord > budget:
                log(
                    f"{tag}: ordering budget spent at {done}/{len(jobs)}; the rest keep file order"
                )
                break
    if ORDER_CACHE and done:
        import json as _json

        _t = Path(ORDER_CACHE).with_suffix(".tmp")
        _t.write_text(_json.dumps(seen))
        _t.replace(Path(ORDER_CACHE))
    log(
        f"{tag}: ordered {ok}/{n_job} by geometry ({n_job - ok} kept arbitrary) in {time.time() - t_ord:.0f}s"
    )
    jobs = [
        (st, k, plane, slot_map[st][name])
        for st in studies
        for k, (name, plane, _, _) in enumerate(SLOTS)
        if name in slot_map[st]
    ]
    log(f"{tag}: decoding {len(jobs)} slot-series")
    n_failed_before = len(DECODE_FAILED)
    CHUNK = 512
    done = 0
    with ThreadPoolExecutor(max_workers=PIX_THREADS) as pool:
        for c0 in range(0, len(jobs), CHUNK):
            block = jobs[c0 : c0 + CHUNK]
            for (st, k, plane, _), img in zip(
                block, pool.map(lambda j: read_slot(j[3], CACHE_SLICES, IMG), block)
            ):
                done += 1
                if img is None:
                    continue
                cache[sidx[st], k] = normalise_laterality(
                    img, plane, lat_map.get(st)
                ).numpy()
                mask[sidx[st], k] = 1.0
            if done % 4096 < CHUNK:
                log(f"  {tag} {done}/{len(jobs)}")
            if time.time() - T0 > TIME_BUDGET:
                log(f"  {tag}: time budget reached during decode")
                break
    n_failed = len(DECODE_FAILED) - n_failed_before
    log(
        f"{tag}: {int(mask.sum())}/{len(jobs)} slots filled"
        + (f"; {n_failed} series had a slice that would not decode" if n_failed else "")
    )
    gc.collect()
    return (studies, cache, mask)


SLOTS = [
    ("SAG_FS", "Sagittal", None, True),
    ("COR_FS", "Coronal", None, True),
    ("AX_FS", "Axial", None, True),
]


N_SLOT = len(SLOTS)


CACHE_SLICES = 12


TIME_BUDGET = 8.72 * 3600


IMG = CACHE_IMG = 336


CROP_MM = 10_000.0


SLICE_BAND = (0.12, 0.88)


RULES = dict(RULES_LEGACY)


TOKEN_DIM = 1024


HEAD_DIM = 512


KERAS_BGR_MEAN = (103.939, 116.779, 123.68)


KERAS_BATCHNORM_EPSILON = 1.001e-5


LIMIT_CORPUS = None


def find_input_file(name):
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if name in files:
            return Path(root) / name
    raise FileNotFoundError(name)


def make_targets(train):
    """Three independent public report teachers; image-read gold always wins."""
    uid = "StudyInstanceUID"
    sources = [
        pd.read_csv(find_input_file("report_labels_v2.csv")),
        pd.read_csv(find_input_file("llm_labels_v2.csv")),
        pd.read_csv(find_input_file("labels_llm_gpt56sol.csv")),
    ]
    cube = []
    for frame in sources:
        if frame[uid].duplicated().any():
            raise ValueError("duplicate study in report-label source")
        aligned = train[[uid]].merge(frame[[uid] + TARGETS], on=uid, how="left")
        cube.append(aligned[TARGETS].to_numpy(float))
    cube = np.stack(cube)
    available = np.isfinite(cube).sum(0)
    if np.any(available < 2):
        raise ValueError("fewer than two report teachers for a study/target")
    y = np.nanmean(cube, axis=0).astype(np.float32)
    disagreement = np.nanmean(np.abs(cube - y[None]), axis=0)
    agreement = np.clip(1.0 - 2.0 * disagreement, 0, 1)
    certainty = np.clip(2.0 * np.abs(y - 0.5), 0, 1)
    w = (0.15 + 0.85 * (0.65 * agreement + 0.35 * certainty)).astype(np.float32)
    gold = train[TARGETS].notna().all(axis=1).to_numpy()
    y[gold] = train.loc[gold, TARGETS].to_numpy(np.float32)
    w[gold] = 3.0
    return y, w, gold


def _v52_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode(n):
    if isinstance(n, bytes):
        return n.decode()
    else:
        return n


def keras_to_torch_densenet(path):
    state = {}
    with h5py.File(path, "r") as handle:
        weights = handle["model_weights"]
        for layer in map(decode, weights.attrs["layer_names"]):  # type: ignore
            names = list(map(decode, weights[layer].attrs.get("weight_names", [])))  # type: ignore
            if not names:
                continue
            w = {
                n.split("/")[-1].split(":")[0]: np.asarray(weights[layer][n])  # type: ignore
                for n in names
            }
            if layer == "conv1/conv":
                key = "conv0"
            elif layer == "conv1/bn":
                key = "norm0"
            elif layer == "bn":
                key = "norm5"
            elif m := re.fullmatch(r"conv(\d)_block(\d+)_(\d)_(bn|conv)", layer):
                sub = {
                    ("0", "bn"): "norm1",
                    ("1", "conv"): "conv1",
                    ("1", "bn"): "norm2",
                    ("2", "conv"): "conv2",
                }[(m[3], m[4])]
                key = f"denseblock{int(m[1]) - 1}.denselayer{int(m[2])}.{sub}"
            elif m := re.fullmatch(r"pool(\d)_(bn|conv)", layer):
                key = f"transition{int(m[1]) - 1}.{'norm' if m[2] == 'bn' else 'conv'}"
            else:
                raise RuntimeError(f"unmapped Keras DenseNet layer {layer}")
            prefix = f"backbone.0.{key}."
            if "kernel" in w:
                state[prefix + "weight"] = (
                    torch.from_numpy(w["kernel"]).permute(3, 2, 0, 1).contiguous()
                )
            else:
                state[prefix + "weight"] = torch.from_numpy(w["gamma"])
                state[prefix + "bias"] = torch.from_numpy(w["beta"])
                state[prefix + "running_mean"] = torch.from_numpy(w["moving_mean"])
                state[prefix + "running_var"] = torch.from_numpy(w["moving_variance"])
                state[prefix + "num_batches_tracked"] = torch.tensor(0)
    return state


def load_radimagenet(device):
    """Strictly load the RadImageNet DenseNet-121 checkpoint."""
    from torchvision.models import densenet121

    checkpoint = find_input_file("RadImageNet-DenseNet121_notop.h5")
    expected_checkpoint = (
        "5266539752a33d3345b1e4690ddd7bef80a2f0e2b534a4ece4ff75625abdb976"
    )
    observed_checkpoint = _v52_sha256(checkpoint)
    if observed_checkpoint != expected_checkpoint:
        raise RuntimeError(f"RadImageNet checkpoint drift: {observed_checkpoint}")

    class RadImageNetEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                *list(densenet121(weights=None).children())[:-1]
            )
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eps = KERAS_BATCHNORM_EPSILON

        def forward(self, image):
            return F.relu(self.backbone(image)).mean(dim=(2, 3))

    model = RadImageNetEncoder()
    state = keras_to_torch_densenet(checkpoint)
    if not state or not all(str(key).startswith("backbone.") for key in state):
        raise RuntimeError("unexpected RadImageNet state-dict namespace")
    model.load_state_dict(state, strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 6953856:
        raise RuntimeError(f"unexpected RadImageNet parameter count {parameter_count}")
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
    if gpu_count > 1:
        model = nn.DataParallel(model, device_ids=list(range(gpu_count)))
    log(
        f"RadImageNet strict load: {parameter_count:,} params; "
        f"inference GPUs={max(1, gpu_count)}"
    )
    return model


@torch.inference_mode()
def encode_radimagenet(cache, slot_mask, device):
    """Encode acquired slices with the Keras caffe-mode RadImageNet contract."""
    n, slots, slices, h, w = cache.shape
    features = np.zeros((n, slots * slices, TOKEN_DIM), np.float16)
    token_mask = np.repeat(slot_mask[:, :, None], slices, axis=2).reshape(n, -1)
    valid = np.flatnonzero(token_mask.reshape(-1) > 0)
    flat = cache.reshape(-1, h, w)
    model = load_radimagenet(device)
    if device.type == "cuda":
        batch = 192 if torch.cuda.device_count() > 1 else 96
    else:
        batch = 8
    mean = torch.tensor(KERAS_BGR_MEAN, device=device).view(1, 3, 1, 1)
    for b0 in range(0, len(valid), batch):
        ix = valid[b0 : b0 + batch]
        x = torch.from_numpy(flat[ix]).to(device).float().unsqueeze(1)
        x = (x - mean) / 255.0
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            feat = model(x)
        if feat.shape[1:] != (TOKEN_DIM,):
            raise RuntimeError(
                f"unexpected RadImageNet feature shape {tuple(feat.shape)}"
            )
        features.reshape(-1, TOKEN_DIM)[ix] = (
            feat.float().cpu().numpy().astype(np.float16)
        )
        if b0 % (batch * 100) == 0:
            log(f"RadImageNet encoded {b0}/{len(valid)} acquired slices")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return features, token_mask.astype(np.float32)


class FoundationQueryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(
            nn.LayerNorm(TOKEN_DIM), nn.Linear(TOKEN_DIM, HEAD_DIM), nn.GELU()
        )
        self.plane = nn.Parameter(torch.randn(N_SLOT, HEAD_DIM) * 0.01)
        self.position = nn.Parameter(torch.randn(CACHE_SLICES, HEAD_DIM) * 0.01)
        self.query = nn.Parameter(torch.randn(len(TARGETS), HEAD_DIM) * 0.02)
        self.attn = nn.MultiheadAttention(HEAD_DIM, 8, dropout=0.10, batch_first=True)
        self.fuse = nn.Sequential(
            nn.LayerNorm(HEAD_DIM * 4),
            nn.Linear(HEAD_DIM * 4, HEAD_DIM),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.weight = nn.Parameter(torch.randn(len(TARGETS), HEAD_DIM) * 0.02)
        self.bias = nn.Parameter(torch.zeros(len(TARGETS)))

    def forward(self, feature, mask):
        token = self.project(feature.float())
        token = token.view(len(token), N_SLOT, CACHE_SLICES, HEAD_DIM)
        token = token + self.plane[None, :, None] + self.position[None, None]
        token = token.flatten(1, 2)
        key_padding = mask <= 0
        # No study should be empty, but keep MHA numerically defined if one is.
        all_empty = key_padding.all(1)
        if all_empty.any():
            key_padding = key_padding.clone()
            key_padding[all_empty, 0] = False
        query = self.query.unsqueeze(0).expand(len(token), -1, -1)
        attended = (
            query
            + self.attn(
                query, token, token, key_padding_mask=key_padding, need_weights=False
            )[0]
        )
        denom = mask.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
        mean = (token * mask.unsqueeze(-1)).sum(1, keepdims=True) / denom
        mean = mean.expand(-1, len(TARGETS), -1)
        fused = self.fuse(
            torch.cat([attended, mean, torch.abs(attended - mean), attended * mean], -1)
        )
        return (fused * self.weight.unsqueeze(0)).sum(-1) + self.bias


def macro_auc(y, pred):
    from sklearn.metrics import roc_auc_score

    hard = (np.asarray(y) >= 0.5).astype(np.uint8)
    values = [
        roc_auc_score(hard[:, j], pred[:, j])
        for j in range(hard.shape[1])
        if np.unique(hard[:, j]).size == 2
    ]
    return float(np.mean(values))


@torch.inference_mode()
def predict_head(model, features, masks, indices, device, batch=64):
    model.eval()
    pred = []
    for b0 in range(0, len(indices), batch):
        ix = indices[b0 : b0 + batch]
        x = torch.from_numpy(features[ix]).to(device)
        m = torch.from_numpy(masks[ix]).to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            pred.append(torch.sigmoid(model(x, m)).float().cpu())
    return torch.cat(pred).numpy()


def train_fold(features, masks, y, weights, train_idx, val_idx, fold, device):
    from torch.utils.data import DataLoader, Dataset

    class Rows(Dataset):
        def __init__(self, indices):
            self.indices = np.asarray(indices)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, k):
            i = self.indices[k]
            return features[i], masks[i], y[i], weights[i]

    model = FoundationQueryHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=3e-3)
    generator = torch.Generator().manual_seed(SEED + 100 + fold)
    loader = DataLoader(
        Rows(train_idx),
        batch_size=48,
        shuffle=True,
        generator=generator,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    best, best_auc, stale = None, -1.0, 0
    for epoch in range(24):
        model.train()
        for x, m, target, weight in loader:
            x, m = x.to(device), m.to(device)
            target, weight = target.to(device), weight.to(device)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x, m)
                raw = F.binary_cross_entropy_with_logits(
                    logits, target, reduction="none"
                )
                loss = (raw * weight).sum() / weight.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        pred = predict_head(model, features, masks, val_idx, device)
        score = macro_auc(y[val_idx], pred)
        log(f"fold {fold} epoch {epoch}: grouped weak-val AUC {score:.5f}")
        if score > best_auc + 2e-4:
            best_auc, stale = score, 0
            best = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 5:
                break
    return best, best_auc


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output = Path(f"/kaggle/working/densenet_rad_{IMG}x{CACHE_SLICES}")
    output.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(ROOT / "train.csv", dtype={"StudyInstanceUID": str})
    if LIMIT_CORPUS:
        train = train.iloc[:LIMIT_CORPUS].reset_index(drop=True)
        log(f"LIMIT_STUDIES={LIMIT_CORPUS}: smoke-test run on a subset")
    train_series = pd.read_csv(
        ROOT / "train_series.csv",
        dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
    )
    plane = dict(zip(train_series.SeriesInstanceUID, train_series.Anatomical_Plane))
    headers = annotate(walk("train_series"))
    headers = headers[headers.StudyInstanceUID.isin(set(train.StudyInstanceUID))]
    studies, pixels, slot_mask = build_cache(
        pick_slots(headers, plane), plane, lat_of(headers, "train "), "train"
    )
    by_uid = {str(uid): i for i, uid in enumerate(studies)}
    order = np.array([by_uid[uid] for uid in train.StudyInstanceUID], dtype=np.int64)
    features, token_mask = encode_radimagenet(pixels, slot_mask, device)
    del pixels, slot_mask, headers
    gc.collect()
    features, token_mask = features[order], token_mask[order]
    np.savez(
        output / "features.npz",
        features=features,
        token_mask=token_mask,
        study_ids=train.StudyInstanceUID.to_numpy(str),
    )
    log(f"saved encoder features {features.shape} to {output / 'features.npz'}")

    y, weights, gold = make_targets(train)
    reference = pd.read_csv(
        find_input_file("v52_oof.csv"), dtype={"StudyInstanceUID": str}
    ).set_index("StudyInstanceUID")
    fold_id = reference.loc[train.StudyInstanceUID, "fold"].to_numpy()

    oof = np.zeros_like(y, dtype=np.float32)
    heads = []
    for fold in range(5):
        tr, va = np.flatnonzero(fold_id != fold), np.flatnonzero(fold_id == fold)
        state, score = train_fold(
            features, token_mask, y, weights, tr, va, fold, device
        )
        if state is None:
            raise RuntimeError(f"fold {fold} produced no checkpoint")
        head = FoundationQueryHead().to(device)
        head.load_state_dict(state, strict=True)
        oof[va] = predict_head(head, features, token_mask, va, device)
        heads.append({"fold": fold, "weak_auc": float(score), "state_dict": state})
        del head
        torch.cuda.empty_cache()

    log(f"OOF weak macro AUC {macro_auc(y, oof):.5f}")
    log(
        f"OOF gold macro AUC {macro_auc(y[gold], oof[gold]):.5f} on {int(gold.sum())} studies"
    )

    torch.save(
        {
            "targets": TARGETS,
            "slots": SLOTS,
            "img": IMG,
            "slices_per_plane": CACHE_SLICES,
            "crop_mm": CROP_MM,
            "folds": heads,
        },
        output / "heads.pt",
    )
    frame = pd.DataFrame(oof, columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", train.StudyInstanceUID)
    frame["fold"] = fold_id
    frame["is_gold"] = gold.astype(np.uint8)
    frame.to_csv(output / "oof.csv", index=False)


if __name__ == "__main__":
    main()
