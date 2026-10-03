import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import densenet121


DATA = Path(__file__).resolve().parent.parent / "data"
CORPUS_DIR = DATA / "raptor_corpus"
OUTPUT = DATA / "raptor_densenet_runs"
DENSENET_H5 = DATA / "weights" / "RadImageNet-DenseNet121_notop.h5"
SOFT_LABELS = DATA / "labels" / "labels_llm_soft.parquet"
AGREEMENT_WEIGHTS = DATA / "labels" / "labels_llm_weights.parquet"
FOLD_FILE = DATA / "labels" / "folds.json"
TRAIN_CSV = DATA / "competition" / "train.csv"

LAB = [
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
TOKEN_DIM = 1024
KERAS_BGR_MEAN = (103.939, 116.779, 123.68)
KERAS_BATCHNORM_EPSILON = 1.001e-5


def find_file(local, name):
    if Path(local).exists():
        return str(local)
    for d, _, fs in os.walk("/kaggle/input"):
        if name in fs:
            return os.path.join(d, name)
    raise SystemExit(f"could not find {name} at {local} or under /kaggle/input")


class TwoPartVols:
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.n_a = a.shape[0]
        self.shape = (a.shape[0] + b.shape[0],) + tuple(a.shape[1:])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, row):
        return self.a[row] if row < self.n_a else self.b[row - self.n_a]


def open_corpus(corpus_dir):
    corpus_dir = Path(corpus_dir)
    vols = TwoPartVols(
        np.load(find_file(corpus_dir / "all_vols.npy", "all_vols.npy"), mmap_mode="r"),
        np.load(
            find_file(corpus_dir / "extra_vols.npy", "extra_vols.npy"), mmap_mode="r"
        ),
    )
    masks = np.concatenate(
        [
            np.load(find_file(corpus_dir / "all_masks.npy", "all_masks.npy")),
            np.load(find_file(corpus_dir / "extra_masks.npy", "extra_masks.npy")),
        ],
        axis=0,
    )
    return vols, masks


def open_ids(corpus_dir):
    corpus_dir = Path(corpus_dir)
    return np.concatenate(
        [
            np.load(
                find_file(corpus_dir / "all_ids.npy", "all_ids.npy"), allow_pickle=True
            ).astype(str),
            np.load(
                find_file(corpus_dir / "extra_ids.npy", "extra_ids.npy"),
                allow_pickle=True,
            ).astype(str),
        ]
    )


class StudyWindows(Dataset):
    def __init__(
        self, corpus_dir, ids, id2row, labels, res, k, train, aug=True, weights=None
    ):
        self.corpus_dir = corpus_dir
        self.ids = ids
        self.id2row = id2row
        self.labels = labels
        self.weights = weights or {}
        self.res, self.k, self.train, self.aug = res, k, train, aug
        self.vols = None
        self.masks = None

    def __len__(self):
        return len(self.ids)

    def _ensure(self):
        if self.vols is None:
            self.vols, self.masks = open_corpus(self.corpus_dir)

    def _centers(self, valid, count):
        lo, hi = int(valid.min()), int(valid.max())
        cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
        if not cs:
            cs = [max(1, min((lo + hi) // 2, self._D - 2))]
        if self.train:
            reps = count // len(cs) + 1
            pool = cs * reps
            random.shuffle(pool)
            return pool[:count]
        idx = np.linspace(0, len(cs) - 1, count).round().astype(int)
        return [cs[i] for i in idx]

    def _resize(self, tri):
        t = torch.from_numpy(np.ascontiguousarray(tri))
        if t.shape[-1] != self.res or t.shape[-2] != self.res:
            t = F.interpolate(
                t[None], size=(self.res, self.res), mode="bilinear", align_corners=False
            )[0]
        return t.numpy()

    def __getitem__(self, i):
        self._ensure()
        uid = self.ids[i]
        row = self.id2row[uid]
        self._D = self.vols.shape[1]
        valid = np.where(self.masks[row] > 0)[0]
        if len(valid) < 3:
            valid = np.arange(min(3, self._D))
        cs = self._centers(valid, self.k)
        vol = self.vols[row]
        tiles = []
        for c in cs:
            c = max(1, min(c, self._D - 2))
            tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32)
            tiles.append(self._resize(tri / 255.0))
        x = torch.from_numpy(np.stack(tiles, 0))
        if self.train and self.aug:
            g = 1.0 + (random.random() - 0.5) * 0.20
            x = (x * g).clamp(0, 1)
        w = self.weights.get(uid)
        if w is None:
            w = np.ones(len(LAB), np.float32)
        return x, torch.from_numpy(self.labels[uid]), torch.from_numpy(w)


def collate(batch):
    return tuple(torch.stack([b[i] for b in batch]) for i in range(3))


def geometric_augment(x, rotate_deg, scale, shift):
    b, k, c, h, w = x.shape
    flat = x.flatten(0, 1)
    n, device = flat.shape[0], flat.device
    angle = (torch.rand(n, device=device) * 2 - 1) * math.radians(rotate_deg)
    zoom = 1 + (torch.rand(n, device=device) * 2 - 1) * scale
    offset = (torch.rand(n, 2, device=device) * 2 - 1) * shift
    cos, sin = torch.cos(angle) / zoom, torch.sin(angle) / zoom
    theta = torch.stack(
        [
            torch.stack([cos, -sin, offset[:, 0]], 1),
            torch.stack([sin, cos, offset[:, 1]], 1),
        ],
        1,
    )
    grid = F.affine_grid(theta, list(flat.shape), align_corners=False)
    out = F.grid_sample(
        flat, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    return out.view(b, k, c, h, w)


def decode(n):
    return n.decode() if isinstance(n, bytes) else n


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
            prefix = f"{key}."
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


class RadImageNetBackbone(nn.Module):
    num_features = TOKEN_DIM

    def __init__(self, weights, grad_ckpt):
        super().__init__()
        self.features = densenet121(weights=None, memory_efficient=grad_ckpt).features
        for module in self.features.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eps = KERAS_BATCHNORM_EPSILON
        if weights:
            self.features.load_state_dict(keras_to_torch_densenet(weights), strict=True)
        self.register_buffer(
            "mean", torch.tensor(KERAS_BGR_MEAN).view(1, 3, 1, 1) / 255.0
        )

    def forward(self, x):
        return F.relu(self.features(x - self.mean)).mean(dim=(2, 3))


class RaptorClassifier(nn.Module):
    def __init__(self, backbone, F_dim, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(
            nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n)
        )
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x):
        B, K = x.shape[:2]
        return self.backbone(x.flatten(0, 1)).view(B, K, -1)

    def head(self, feats):
        h = self.norm(feats)
        a = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.clsW).sum(-1) + self.clsb

    def forward(self, x):
        return self.head(self.encode(x))


def macro_auc(truth, pred):
    hard = (np.asarray(truth) >= 0.5).astype(int)
    aucs = {}
    for j, name in enumerate(LAB):
        if len(set(hard[:, j])) > 1:
            aucs[name] = float(roc_auc_score(hard[:, j], pred[:, j]))
    return float(np.mean(list(aucs.values()))), aucs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DENSENET_H5))
    ap.add_argument("--corpus_dir", default=str(CORPUS_DIR))
    ap.add_argument("--output", default=str(OUTPUT))
    ap.add_argument("--labels", default=str(SOFT_LABELS))
    ap.add_argument("--train_csv", default=str(TRAIN_CSV))
    ap.add_argument("--res", type=int, default=336)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--k_eval", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--bb_lr", type=float, default=3e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--random_init", action="store_true")
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--fold_file", default=str(FOLD_FILE))
    ap.add_argument("--geo_aug", action="store_true")
    ap.add_argument("--rotate_deg", type=float, default=10.0)
    ap.add_argument("--scale", type=float, default=0.10)
    ap.add_argument("--shift", type=float, default=0.05)
    ap.add_argument("--agreement_weights", action="store_true")
    ap.add_argument("--agreement_file", default=str(AGREEMENT_WEIGHTS))
    ap.add_argument("--train_eval", type=int, default=500)
    ap.add_argument("--ema", type=float, default=0.0)
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.smoke:
        a.epochs, a.bs, a.k, a.k_eval, a.limit, a.workers = 2, 4, 4, 6, 40, 0
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"device {dev} | res {a.res} | k {a.k}/{a.k_eval} | bs {a.bs} | tag {a.tag} | "
        f"geo_aug {a.geo_aug} | agreement_weights {a.agreement_weights}",
        flush=True,
    )

    ids = open_ids(a.corpus_dir)
    id2row = {u: i for i, u in enumerate(ids)}
    idset = set(ids)
    tr = pd.read_csv(find_file(a.train_csv, "train.csv"), dtype={"StudyInstanceUID": str})
    gold_df = tr[tr[LAB].notna().all(axis=1)].copy().set_index("StudyInstanceUID")
    gold_ids = [u for u in gold_df.index if u in idset]
    soft = pd.read_parquet(find_file(a.labels, "labels_llm_soft.parquet"))
    soft["StudyInstanceUID"] = soft["StudyInstanceUID"].astype(str)
    soft = soft.set_index("StudyInstanceUID")
    goldset = set(gold_ids)
    train_ids = [u for u in ids if u in soft.index and u not in goldset]
    oof_ids = []
    if a.folds > 0:
        assert 0 <= a.fold < a.folds, f"--fold must be in [0,{a.folds}) when --folds>0"
        fmap = json.load(open(a.fold_file))["folds"]
        held = set(u for u in train_ids if fmap.get(u, -1) == a.fold)
        oof_ids = [u for u in train_ids if u in held]
        train_ids = [u for u in train_ids if u not in held]
    if a.limit:
        train_ids = train_ids[: a.limit]
        oof_ids = oof_ids[: a.limit]
    labels = {u: soft.loc[u, LAB].values.astype(np.float32) for u in train_ids}
    for u in oof_ids:
        labels[u] = soft.loc[u, LAB].values.astype(np.float32)
    for u in gold_ids:
        labels[u] = gold_df.loc[u, LAB].values.astype(np.float32)
    print(
        f"train {len(train_ids)} | gold-val {len(gold_ids)} | fold {a.fold} oof {len(oof_ids)}",
        flush=True,
    )

    label_weights = {}
    if a.agreement_weights:
        aw = pd.read_parquet(find_file(a.agreement_file, "labels_llm_weights.parquet"))
        aw["StudyInstanceUID"] = aw["StudyInstanceUID"].astype(str)
        aw = aw.set_index("StudyInstanceUID")
        label_weights = {u: aw.loc[u, LAB].values.astype(np.float32) for u in train_ids}
        mean_w = np.stack(list(label_weights.values())).mean(0)
        print(
            "mean agreement weight per finding: "
            + ", ".join(f"{t} {v:.2f}" for t, v in zip(LAB, mean_w)),
            flush=True,
        )

    prev = np.clip(np.stack([labels[u] for u in train_ids]).mean(0), 0.03, 0.7)
    pw = torch.tensor(
        np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32, device=dev
    )

    def windows(study_ids, k, train):
        return StudyWindows(
            a.corpus_dir,
            study_ids,
            id2row,
            labels,
            a.res,
            k,
            train,
            weights=label_weights if train else None,
        )

    def eval_loader(study_ids, generator=None):
        return DataLoader(
            windows(study_ids, a.k_eval, False),
            batch_size=max(2, a.bs // 2),
            shuffle=False,
            num_workers=a.workers,
            collate_fn=collate,
            persistent_workers=a.workers > 0,
            generator=generator,
        )

    tl = DataLoader(
        windows(train_ids, a.k, True),
        batch_size=a.bs,
        shuffle=True,
        num_workers=a.workers,
        drop_last=True,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=a.workers > 0,
    )
    vl = eval_loader(gold_ids)
    ol = eval_loader(oof_ids) if oof_ids else None
    train_eval_ids = []
    if a.train_eval > 0:
        train_eval_ids = [
            str(u)
            for u in np.random.default_rng(a.seed + a.fold).choice(
                train_ids, size=min(a.train_eval, len(train_ids)), replace=False
            )
        ]
    tel = (
        eval_loader(train_eval_ids, torch.Generator().manual_seed(a.seed))
        if train_eval_ids
        else None
    )

    weights = None if a.random_init else find_file(a.weights, Path(a.weights).name)
    bb = RadImageNetBackbone(weights, a.grad_ckpt)
    src = "random-init" if a.random_init else Path(a.weights).name
    model = RaptorClassifier(bb, F_dim=bb.num_features).to(dev)
    n_bb = sum(p.numel() for p in model.backbone.parameters())
    n_head = sum(p.numel() for p in model.parameters()) - n_bb
    print(
        f"backbone {src} {n_bb:,} params | head {n_head:,} params | grad_ckpt {a.grad_ckpt}",
        flush=True,
    )

    ema_float, live_float, ema_other, live_other = [], [], [], []
    if a.ema > 0:
        for v in model.state_dict().values():
            if v.is_floating_point():
                live_float.append(v)
                ema_float.append(v.detach().clone())
            else:
                live_other.append(v)
                ema_other.append(v.detach().clone())

    head_params = [
        p for n_, p in model.named_parameters() if not n_.startswith("backbone.")
    ]
    bb_params = [p for n_, p in model.named_parameters() if n_.startswith("backbone.")]
    opt = torch.optim.AdamW(
        [
            {"params": bb_params, "lr": a.bb_lr},
            {"params": head_params, "lr": a.head_lr},
        ],
        weight_decay=a.wd,
    )
    steps = max(len(tl) * a.epochs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[a.bb_lr, a.head_lr], total_steps=steps, pct_start=0.15
    )
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)

    @torch.no_grad()
    def predict(loader):
        model.eval()
        P, Y = [], []
        for x, y, _ in loader:
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                o = torch.sigmoid(model(x.to(dev)).float())
            P.append(o.cpu().numpy())
            Y.append(y.numpy())
        return np.concatenate(P), np.concatenate(Y)

    def snapshot(gold_auc, aucs, epoch):
        return {
            "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "gold_auc": gold_auc,
            "aucs": aucs,
            "src": src,
            "res": a.res,
            "arch": "densenet121_radimagenet",
            "lab": LAB,
            "epoch": epoch,
        }

    def save_variant(suffix, state, gold_pred, gold_truth, oof_pred, oof_truth):
        torch.save(state, out / f"raptor_ft_{a.tag}{suffix}.pt")
        np.savez(
            out / f"raptor_gold_{a.tag}{suffix}.npz",
            pred=gold_pred,
            truth=gold_truth,
            ids=np.array(gold_ids),
        )
        if oof_pred is not None:
            np.savez(
                out / f"raptor_oof_{a.tag}{suffix}_fold{a.fold}.npz",
                pred=oof_pred,
                truth=oof_truth,
                ids=np.array(oof_ids),
                fold=a.fold,
                nfolds=a.folds,
            )

    best, best_weak, t0, hist, topk = 0.0, 0.0, time.time(), [], []
    picked = {}
    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        for x, y, w in tl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            if a.geo_aug:
                x = geometric_augment(x, a.rotate_deg, a.scale, a.shift)
            opt.zero_grad()
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits = model(x)
            if a.agreement_weights:
                w = w.to(dev, non_blocking=True)
                raw = F.binary_cross_entropy_with_logits(
                    logits.float(), y, pos_weight=pw, reduction="none"
                )
                loss = (raw * w).sum() / w.sum().clamp_min(1e-6)
            else:
                loss = lossf(logits.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            sched.step()
            if ema_float:
                with torch.no_grad():
                    torch._foreach_lerp_(ema_float, live_float, 1 - a.ema)
                    for e, v in zip(ema_other, live_other):
                        e.copy_(v)
            tot += loss.item()
        P, Y = predict(vl)
        au, aucs = macro_auc(Y, P)
        row = {"ep": ep, "loss": tot / len(tl), "gold_auc": au}
        Po = Yo = weak = None
        extra = ""
        if ol is not None:
            Po, Yo = predict(ol)
            weak = macro_auc(Yo, Po)[0]
            row["oof_weak_auc"] = weak
        if tel is not None:
            Pt, Yt = predict(tel)
            row["train_weak_auc"] = macro_auc(Yt, Pt)[0]
            extra = f" | train AUC {row['train_weak_auc']:.5f}"
        hist.append(row)
        state = snapshot(au, aucs, ep)
        if au > best:
            best = au
            picked["best gold"] = {"epoch": ep, "gold": au, "weak": weak, "aucs": aucs}
            save_variant("", state, P, Y, Po, Yo)
        if weak is not None and weak > best_weak:
            best_weak = weak
            picked["best weak"] = {"epoch": ep, "gold": au, "weak": weak}
            save_variant("_bestweak", state, P, Y, Po, Yo)
        if weak is not None:
            extra = (
                f" | fold-{a.fold} weak AUC {weak:.5f} (best {best_weak:.5f})" + extra
            )
        if len(topk) < max(1, a.topk) or au > min(t["gold_auc"] for t in topk):
            topk.append(state)
            topk.sort(key=lambda t: -t["gold_auc"])
            del topk[max(1, a.topk) :]
        json.dump(
            {
                "tag": a.tag,
                "src": src,
                "best_gold_auc": best,
                "best_weak_auc": best_weak,
                "hist": hist,
                "args": vars(a),
                "n_train": len(train_ids),
                "n_gold": len(gold_ids),
                "epochs_done": ep + 1,
            },
            open(out / f"raptor_ft_{a.tag}.json", "w"),
            indent=1,
        )
        print(
            f"ep{ep} loss {tot / len(tl):.3f} | gold AUC {au:.4f} (best {best:.4f})"
            f"{extra} | {time.time() - t0:.0f}s",
            flush=True,
        )
    picked["last"] = {"epoch": ep, "gold": au, "weak": weak}
    save_variant("_last", state, P, Y, Po, Yo)

    print(f"\nDONE {src}")
    for k, v in picked["best gold"]["aucs"].items():
        print(f"   {k:18s} {v:.3f}", flush=True)
    for name, r in picked.items():
        line = f"{name:10s} epoch {r['epoch']:>2} | gold AUC {r['gold']:.4f}"
        if r["weak"] is not None:
            line += f" | fold-{a.fold} weak AUC {r['weak']:.5f}"
        print(line, flush=True)

    if len(topk) > 1:
        sds = [t["model"] for t in topk]
        avg = {}
        for k in sds[0]:
            v0 = sds[0][k]
            if v0.is_floating_point():
                avg[k] = sum(sd[k].double() for sd in sds).div(len(sds)).to(v0.dtype)
            else:
                avg[k] = v0.clone()
        model.load_state_dict(avg)
        P, Y = predict(vl)
        au_swa, aucs_swa = macro_auc(Y, P)
        epochs = [int(t["epoch"]) for t in topk]
        swa_state = snapshot(au_swa, aucs_swa, epochs)
        swa_state["model"] = avg
        Po = Yo = None
        line = f"weight average over epochs {epochs} | gold AUC {au_swa:.4f}"
        if ol is not None:
            Po, Yo = predict(ol)
            line += f" | fold-{a.fold} weak AUC {macro_auc(Yo, Po)[0]:.5f}"
        save_variant("_swa", swa_state, P, Y, Po, Yo)
        print(line, flush=True)

    if ema_float:
        floats = iter(ema_float)
        others = iter(ema_other)
        ema_sd = {
            k: (next(floats) if v.is_floating_point() else next(others))
            for k, v in model.state_dict().items()
        }
        model.load_state_dict(ema_sd)
        P, Y = predict(vl)
        au_ema, aucs_ema = macro_auc(Y, P)
        Po = Yo = None
        line = f"EMA (decay {a.ema}) | gold AUC {au_ema:.4f}"
        if ol is not None:
            Po, Yo = predict(ol)
            line += f" | fold-{a.fold} weak AUC {macro_auc(Yo, Po)[0]:.5f}"
        save_variant("_ema", snapshot(au_ema, aucs_ema, f"ema{a.ema}"), P, Y, Po, Yo)
        print(line, flush=True)


if __name__ == "__main__":
    main()
