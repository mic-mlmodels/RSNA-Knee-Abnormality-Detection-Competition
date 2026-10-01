import math
import re
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import densenet121


DATA = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = DATA / "pixel_cache"
LABEL_DIR = DATA / "labels"
TRAIN_CSV = DATA / "competition" / "train.csv"
FOLD_SOURCE = DATA / "oof" / "rad_e10_reference_oof.csv"
DENSENET_H5 = DATA / "weights" / "RadImageNet-DenseNet121_notop.h5"
FROZEN_HEADS = DATA / "densenet_run" / "densenet_rad" / "heads.pt"
OUTPUT = DATA / "finetune_runs"

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
SEED = 2026
N_SLOT = 3
CACHE_SLICES = 8
N_TOKENS = N_SLOT * CACHE_SLICES
TOKEN_DIM = 1024
HEAD_DIM = 512
KERAS_BGR_MEAN = (103.939, 116.779, 123.68)
KERAS_BATCHNORM_EPSILON = 1.001e-5

MODES = ("partial",)
FOLDS = (0,)
BATCH_STUDIES = 4
MAX_EPOCHS = 20
PATIENCE = 5
MIN_EPOCHS = 20
TRAIN_EVAL_STUDIES = 500
MIN_DELTA = 2e-4
WARMUP_FRACTION = 0.1
NUM_WORKERS = 4
HEAD_LR = 5e-5
HEAD_WEIGHT_DECAY = 3e-3
ENCODER_WEIGHT_DECAY = 0.0
ENCODER_LR = {
    "partial": {"denseblock4": 2e-5, "norm5": 2e-5},
    "full": {
        "conv0": 2e-6,
        "norm0": 2e-6,
        "denseblock1": 2e-6,
        "transition1": 2e-6,
        "denseblock2": 5e-6,
        "transition2": 5e-6,
        "denseblock3": 1e-5,
        "transition3": 1e-5,
        "denseblock4": 2e-5,
        "norm5": 2e-5,
    },
}
ENCODER_L2SP_ALPHA = 20
HEAD_L2SP_ALPHA = 220
AUG_ROTATE_DEG = 7.0
AUG_SCALE = 0.08
AUG_SHIFT = 0.05
AUG_CONTRAST = 0.10
AUG_BRIGHTNESS = 10.0

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


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


def load_targets(train):
    uid = "StudyInstanceUID"
    sources = [
        pd.read_csv(LABEL_DIR / "report_labels_v2.csv"),
        pd.read_csv(LABEL_DIR / "llm_labels_v2.csv"),
        pd.read_csv(LABEL_DIR / "labels_llm_gpt56sol.csv"),
    ]
    cube = []
    for frame in sources:
        if frame[uid].duplicated().any():
            raise ValueError("duplicate study in report-label source")
        aligned = train[[uid]].merge(frame[[uid] + TARGETS], on=uid, how="left")
        cube.append(aligned[TARGETS].to_numpy(float))
    cube = np.stack(cube)
    if np.any(np.isfinite(cube).sum(0) < 2):
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


class FineTuneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(*list(densenet121(weights=None).children())[:-1])
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eps = KERAS_BATCHNORM_EPSILON
        self.head = FoundationQueryHead()
        self.register_buffer("mean", torch.tensor(KERAS_BGR_MEAN).view(1, 3, 1, 1))

    def forward(self, pixels, mask):
        b, _, h, w = pixels.shape
        x = pixels.reshape(b * N_TOKENS, 1, h, w).float()
        x = (x - self.mean) / 255.0
        feature = F.relu(self.backbone(x)).mean(dim=(2, 3))
        feature = feature.reshape(b, N_TOKENS, TOKEN_DIM) * mask.unsqueeze(-1)
        return self.head(feature, mask)


def build_model(fold, mode, device):
    model = FineTuneModel()
    converted = keras_to_torch_densenet(DENSENET_H5)
    model.backbone.load_state_dict(
        {k.removeprefix("backbone."): v for k, v in converted.items()}, strict=True
    )
    frozen = torch.load(FROZEN_HEADS, map_location="cpu", weights_only=True)
    record = next(r for r in frozen["folds"] if int(r["fold"]) == fold)
    model.head.load_state_dict(record["state_dict"], strict=True)
    for name, parameter in model.backbone.named_parameters():
        parameter.requires_grad_(name.split(".")[1] in ENCODER_LR[mode])
    return model.to(device)


def parameter_groups(model, mode):
    groups = {}
    for name, parameter in model.backbone.named_parameters():
        if parameter.requires_grad:
            groups.setdefault(name.split(".")[1], []).append(parameter)
    result = [
        {
            "params": params,
            "lr": ENCODER_LR[mode][block],
            "weight_decay": ENCODER_WEIGHT_DECAY,
            "l2sp_alpha": ENCODER_L2SP_ALPHA,
        }
        for block, params in groups.items()
    ]
    result.append(
        {
            "params": list(model.head.parameters()),
            "lr": HEAD_LR,
            "weight_decay": HEAD_WEIGHT_DECAY,
            "l2sp_alpha": HEAD_L2SP_ALPHA,
        }
    )
    return result


@torch.no_grad()
def relative_drift(params, old_params):
    diff = sum(((p.float() - a) ** 2).sum() for p, a in zip(params, old_params))
    base = sum((a**2).sum() for a in old_params)
    return float((diff / base).sqrt())  # type: ignore


def set_train_mode(model):
    model.train()
    for module in model.backbone.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def augment(pixels):
    b, _, h, w = pixels.shape
    x = pixels.float().reshape(b * N_SLOT, CACHE_SLICES, h, w)
    n, device = x.shape[0], x.device
    angle = (torch.rand(n, device=device) * 2 - 1) * math.radians(AUG_ROTATE_DEG)
    scale = 1 + (torch.rand(n, device=device) * 2 - 1) * AUG_SCALE
    shift = (torch.rand(n, 2, device=device) * 2 - 1) * AUG_SHIFT
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack(
        [
            torch.stack([cos, -sin, shift[:, 0]], 1),
            torch.stack([sin, cos, shift[:, 1]], 1),
        ],
        1,
    )
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(
        x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    contrast = 1 + (torch.rand(n, 1, 1, 1, device=device) * 2 - 1) * AUG_CONTRAST
    brightness = (torch.rand(n, 1, 1, 1, device=device) * 2 - 1) * AUG_BRIGHTNESS
    x = (x * contrast + brightness).clamp(0, 255)
    return x.reshape(b, N_TOKENS, h, w)


class Studies(Dataset):
    def __init__(self, pixels, mask, y, w, indices):
        self.pixels, self.mask, self.y, self.w = pixels, mask, y, w
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]
        image = np.array(self.pixels[i]).reshape(N_TOKENS, *self.pixels.shape[-2:])
        return torch.from_numpy(image), self.mask[i], self.y[i], self.w[i]


def loader(pixels, mask, y, w, indices, shuffle, batch, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Studies(pixels, mask, y, w, indices),
        batch_size=batch,
        shuffle=shuffle,
        drop_last=shuffle,
        generator=generator,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )


@torch.inference_mode()
def predict(model, pixels, mask, y, w, indices, device):
    model.eval()
    out = []
    for x, m, _, _ in loader(pixels, mask, y, w, indices, False, BATCH_STUDIES * 2):
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(
                x.to(device, non_blocking=True), m.to(device, non_blocking=True)
            )
        out.append(torch.sigmoid(logits.float()).cpu())
    return torch.cat(out).numpy()


def snapshot_params(optimizer):
    return [
        [param.detach().clone() for param in group["params"]]
        for group in optimizer.param_groups
    ]


def apply_weight_decay(optimizer, old_model_params):
    with torch.no_grad():
        for group, old_group in zip(optimizer.param_groups, old_model_params):
            for param, old_param in zip(group["params"], old_group):
                param.sub_(group["lr"] * group["l2sp_alpha"] * (param - old_param))


def train_fold(fold, mode, pixels, mask, y, w, train_idx, val_idx, device, max_epochs):
    model = build_model(fold, mode, device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pred = predict(model, pixels, mask, y, w, val_idx, device)
    best_auc, best_pred, best_state, best_epoch = (
        macro_auc(y[val_idx], pred),
        pred,
        None,
        -1,
    )
    log(
        f"{mode} fold {fold}: {trainable:,} trainable params | start (frozen) val AUC {best_auc:.5f}"
    )
    optimizer = torch.optim.AdamW(parameter_groups(model, mode))
    old_model_params = snapshot_params(optimizer)
    train_loader = loader(
        pixels, mask, y, w, train_idx, True, BATCH_STUDIES, SEED + 100 + fold
    )
    total = max(1, max_epochs * len(train_loader))
    warmup = max(1, int(WARMUP_FRACTION * total))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            min(1.0, (step + 1) / warmup)
            * 0.5
            * (1 + math.cos(math.pi * min(step, total) / total))
        ),
    )
    params = [p for p in model.parameters() if p.requires_grad]
    train_eval_idx = np.random.default_rng(SEED + fold).choice(
        train_idx, size=min(TRAIN_EVAL_STUDIES, len(train_idx)), replace=False
    )
    stale = 0
    for epoch in range(max_epochs):
        set_train_mode(model)
        started, seen, running = time.time(), 0, 0.0
        for x, m, target, weight in train_loader:
            x = augment(x.to(device, non_blocking=True))
            m, target, weight = m.to(device), target.to(device), weight.to(device)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                logits = model(x, m)
            raw = F.binary_cross_entropy_with_logits(
                logits.float(), target, reduction="none"
            )
            loss = (raw * weight).sum() / weight.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            apply_weight_decay(optimizer, old_model_params)
            schedule.step()
            seen += x.shape[0] * N_TOKENS
            running += loss.item()
        pred = predict(model, pixels, mask, y, w, val_idx, device)
        score = macro_auc(y[val_idx], pred)
        train_pred = predict(model, pixels, mask, y, w, train_eval_idx, device)
        train_score = macro_auc(y[train_eval_idx], train_pred)
        rate = seen / max(time.time() - started, 1e-6)
        encoder_drift = relative_drift(
            [p for group in optimizer.param_groups[:-1] for p in group["params"]],
            [a for old_group in old_model_params[:-1] for a in old_group],
        )
        head_drift = relative_drift(
            optimizer.param_groups[-1]["params"], old_model_params[-1]
        )
        log(
            f"{mode} fold {fold} epoch {epoch}: loss {running / max(1, len(train_loader)):.4f} | train AUC {train_score:.5f} | val AUC {score:.5f} | drift enc {encoder_drift:.5f} head {head_drift:.5f} | {rate:.0f} img/s"
        )
        if score > best_auc + MIN_DELTA:
            best_auc, best_pred, best_epoch, stale = score, pred, epoch, 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= PATIENCE and epoch + 1 >= MIN_EPOCHS:
                break
    if best_state is None:
        log(
            f"{mode} fold {fold}: fine-tuning never beat the frozen start; keeping frozen predictions"
        )
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
    del model, optimizer
    torch.cuda.empty_cache()
    return best_state, best_auc, best_epoch, best_pred


def main(
    modes=MODES, folds=FOLDS, max_epochs=MAX_EPOCHS, cache_dir=CACHE_DIR, output=OUTPUT
):
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pixels = np.load(Path(cache_dir) / "pixels.npy", mmap_mode="r")
    slot_mask = np.load(Path(cache_dir) / "slot_mask.npy")
    study_ids = np.load(Path(cache_dir) / "study_ids.npy").astype(str)
    train = pd.read_csv(TRAIN_CSV, dtype={"StudyInstanceUID": str})
    train = train.set_index("StudyInstanceUID").loc[study_ids].reset_index()
    fold_id = pd.read_csv(FOLD_SOURCE, index_col=0).loc[study_ids, "fold"].to_numpy()
    mask = (
        np.repeat(slot_mask[:, :, None], CACHE_SLICES, axis=2)
        .reshape(len(study_ids), -1)
        .astype(np.float32)
    )
    y, w, gold = load_targets(train)
    log(f"pixel cache {pixels.shape} | {len(study_ids)} studies | device {device}")
    for mode in modes:
        target_dir = Path(output) / mode
        target_dir.mkdir(parents=True, exist_ok=True)
        oof = np.full(y.shape, np.nan, dtype=np.float32)
        for fold in folds:
            train_idx, val_idx = (
                np.flatnonzero(fold_id != fold),
                np.flatnonzero(fold_id == fold),
            )
            state, score, epoch, pred = train_fold(
                fold, mode, pixels, mask, y, w, train_idx, val_idx, device, max_epochs
            )
            oof[val_idx] = pred
            torch.save(
                {
                    "fold": fold,
                    "mode": mode,
                    "val_auc": score,
                    "best_epoch": epoch,
                    "state_dict": state,
                },
                target_dir / f"fold{fold}.pt",
            )
            log(f"{mode} fold {fold}: best val AUC {score:.5f} at epoch {epoch}")
        done = ~np.isnan(oof).any(axis=1)
        if done.all():
            log(
                f"{mode}: OOF weak macro AUC {macro_auc(y, oof):.5f} | gold {macro_auc(y[gold], oof[gold]):.5f}"
            )
        frame = pd.DataFrame(oof, columns=TARGETS)
        frame.insert(0, "StudyInstanceUID", study_ids)
        frame["fold"] = fold_id
        frame["is_gold"] = gold.astype(np.uint8)
        frame.to_csv(target_dir / "oof.csv", index=False)


if __name__ == "__main__":
    main()
