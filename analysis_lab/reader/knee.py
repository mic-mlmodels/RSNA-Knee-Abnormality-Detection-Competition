"""Shared dataset and model code for training and inference."""

import math

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

LABELS = [
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
PLANES = ["Sagittal", "Coronal", "Axial"]
# slot = plane index + 3 * (1 - fat_suppressed): fat-suppressed planes first
N_SLOTS = 6
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


N_VIEWS = 5
DEFAULT_VIEW = np.array([[0, 0.0, 0.0]], np.float32)
FULL_VIEW = N_VIEWS
FULL_FOV = 0.92


def slot_of(plane, fat_sup):
    return PLANES.index(plane) + 3 * (1 - int(fat_sup))


def build_study_table(series_df, n_slices):
    """series_df: StudyInstanceUID, SeriesInstanceUID, Fat_Suppression, Anatomical_Plane.
    n_slices: dict series -> cached slice count. Returns {study: [[series ids of slot 0], ...]}
    with each slot's candidates sorted by slice count (largest first)."""
    table = {}
    for st, sr, fs, pl in zip(
        series_df.StudyInstanceUID,
        series_df.SeriesInstanceUID,
        series_df.Fat_Suppression,
        series_df.Anatomical_Plane,
    ):
        if n_slices.get(sr, 0) < 3:
            continue
        table.setdefault(st, [[] for _ in range(N_SLOTS)])[slot_of(pl, fs)].append(sr)
    for slots in table.values():
        for cands in slots:
            cands.sort(key=lambda s: -n_slices[s])
    return table


def window_centres(n, k, train, lo=0.04, hi=0.96):
    """k slice indices (centres of 3-slice windows) spread over the series."""
    a, b = max(1.0, lo * (n - 1)), min(n - 2.0, hi * (n - 1))
    if b < a:
        a = b = (n - 1) / 2
    if train:
        edges = np.linspace(a, b, k + 1)
        c = edges[:-1] + np.random.rand(k) * (edges[1:] - edges[:-1])
    else:
        c = np.linspace(a, b, k)
    return np.clip(np.round(c).astype(int), 1, max(1, n - 2))


class StudyDataset(torch.utils.data.Dataset):
    """One item = one study: uint8 windows [N_SLOTS, k, 3, H, W], slot mask, slice positions, target."""

    def __init__(
        self,
        studies,
        table,
        cache_dir,
        targets=None,
        k=8,
        train=False,
        loader=None,
        views=None,
        passes=3,
        full_p=0.0,
        joint=False,
    ):
        self.studies, self.table, self.cache_dir = list(studies), table, cache_dir
        self.targets, self.k, self.train = targets, k, train
        self.views, self.full_p, self.joint = views, full_p, joint
        self.passes = (
            1 + passes if joint else 1 if train else passes + (1 if full_p > 0 else 0)
        )
        self.loader = loader or (
            lambda s: np.load(f"{self.cache_dir}/{s}.npy", mmap_mode="r")
        )

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        st = self.studies[i]
        slots = self.table.get(st, [[] for _ in range(N_SLOTS)])
        x, mask, pos = (
            None,
            np.zeros(N_SLOTS, bool),
            np.zeros((N_SLOTS, self.k), np.float32),
        )
        ctr = np.zeros((self.passes, N_SLOTS, 2), np.float32)
        vid = np.zeros((self.passes, N_SLOTS), np.int64)
        full_first = self.full_p > 0 and (
            not self.train or np.random.rand() < self.full_p
        )
        for s, cands in enumerate(slots):
            if not cands:
                continue
            sr = cands[np.random.randint(len(cands))] if self.train else cands[0]
            if self.views is not None:
                v = self.views.get(sr, DEFAULT_VIEW)
                if self.joint:
                    vid[:, s] = -1
                    vid[0, s] = FULL_VIEW
                    for j in range(min(len(v), self.passes - 1)):
                        vid[1 + j, s] = int(v[j, 0])
                        ctr[1 + j, s] = v[j, 1:3]
                for p in range(0 if self.joint else self.passes):
                    if full_first and p == 0:
                        vid[p, s] = FULL_VIEW
                        continue
                    q = p - 1 if full_first else p
                    j = np.random.randint(len(v)) if self.train else q % len(v)
                    vid[p, s] = int(v[j, 0])
                    ctr[p, s] = v[j, 1:3]
            vol = self.loader(sr)
            n = len(vol)
            c = window_centres(n, self.k, self.train)
            win = np.stack(
                [vol[np.clip([j - 1, j, j + 1], 0, n - 1)] for j in c]
            )  # [k, 3, H, W]
            if x is None:
                x = np.zeros((N_SLOTS, self.k) + win.shape[1:], np.uint8)
            x[s], mask[s], pos[s] = win, True, c / max(n - 1, 1)
        if x is None:
            x = np.zeros((N_SLOTS, self.k, 3, 384, 384), np.uint8)
        y = (
            self.targets[i]
            if self.targets is not None
            else np.zeros(len(LABELS), np.float32)
        )
        out = (
            torch.from_numpy(x),
            torch.from_numpy(mask),
            torch.from_numpy(pos),
            torch.from_numpy(np.asarray(y, np.float32)),
        )
        if self.views is not None:
            out += (torch.from_numpy(ctr), torch.from_numpy(vid))
        return out


def ghost_views(x, group, p=0.5, ghosts=(2, 8), intensity=(0.2, 0.7), restore=0.02):
    """x: float [N, 3, H, W] in 0..1 with N = n_series * group. Motion ghosting shared within a series:
    every n-th k-space line along one axis is weakened, which repeats the anatomy at multiples of 1/n."""
    out = x.view(-1, group, *x.shape[1:]).clone()
    for i in range(out.shape[0]):
        if np.random.rand() >= p:
            continue
        axis = -2 if np.random.rand() < 0.5 else -1
        n = np.random.randint(ghosts[0], ghosts[1] + 1)
        size = out.shape[axis]
        scale = torch.ones(size, device=x.device)
        scale[::n] = 1 - np.random.uniform(*intensity)
        r = max(1, int(size * restore / 2))
        scale[size // 2 - r : size // 2 + r + 1] = 1
        scale = scale.view(-1, 1) if axis == -2 else scale.view(1, -1)
        kspace = torch.fft.fftshift(torch.fft.fft2(out[i].float()), dim=(-2, -1))
        kspace = torch.fft.ifftshift(kspace * scale, dim=(-2, -1))
        out[i] = torch.fft.ifft2(kspace).abs().to(out.dtype)
    return out.view_as(x)


SAG_SLOTS = [PLANES.index("Sagittal"), PLANES.index("Sagittal") + 3]


def mirror_augment(x, pos, p=0.5):
    m = torch.rand(x.shape[0], device=x.device) < p
    if not m.any():
        return x, pos
    x, pos = x.clone(), pos.clone()
    xs, ps = x[m], pos[m]
    xm = xs.flip(-1)
    xm[:, SAG_SLOTS] = xs[:, SAG_SLOTS].flip(3)
    ps[:, SAG_SLOTS] = 1 - ps[:, SAG_SLOTS]
    x[m], pos[m] = xm, ps
    return x, pos


def make_views(
    x, res, train, group, fov=0.92, ghost=False, centre=None, mean=MEAN, std=STD
):
    """x: uint8 [N, 3, H, W] with N = n_series * group. Crops a centred `fov` fraction of the cached
    field of view and resizes to res; in training adds rotation/scale/shift and gain/gamma jitter
    that is shared within a series."""
    n = x.shape[0]
    ns = n // group
    dev = x.device
    x = x.float().div_(255.0)
    if train:
        ang = (torch.rand(ns, device=dev) - 0.5) * 2 * math.radians(12)
        sc = fov * (1 + (torch.rand(ns, device=dev) - 0.5) * 0.24)
        tx = (torch.rand(ns, device=dev) - 0.5) * 0.12
        ty = (torch.rand(ns, device=dev) - 0.5) * 0.12
        gamma = torch.exp((torch.rand(ns, device=dev) - 0.5) * 0.5)
        gain = 1 + (torch.rand(ns, device=dev) - 0.5) * 0.2
    else:
        ang = torch.zeros(ns, device=dev)
        sc = fov.float() if torch.is_tensor(fov) else torch.full((ns,), fov, device=dev)
        tx = ty = torch.zeros(ns, device=dev)
        gamma = gain = None
    if centre is not None:
        tx = tx + centre[:, 1].to(tx.dtype)
        ty = ty + centre[:, 0].to(ty.dtype)
    cos, sin = torch.cos(ang) * sc, torch.sin(ang) * sc
    theta = torch.stack(
        [torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1
    )
    theta = theta.repeat_interleave(group, 0)
    grid = F.affine_grid(theta, (n, 3, res, res), align_corners=False)
    if x.shape[-1] > 1.5 * res:  # avoid aliasing when shrinking a lot
        x = (
            F.avg_pool2d(x, 2)
            if x.shape[-1] >= 2 * res
            else F.interpolate(x, scale_factor=0.75, mode="bilinear", antialias=True)
        )
    x = F.grid_sample(
        x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    if train:
        g = gamma.repeat_interleave(group).view(n, 1, 1, 1)
        x = x.clamp_min(1e-4).pow(g) * gain.repeat_interleave(group).view(n, 1, 1, 1)
        if ghost:
            x = ghost_views(x, group)
    mean = torch.tensor(mean, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(std, device=dev).view(1, 3, 1, 1)
    return (x - mean) / std


class KneeNet(nn.Module):
    """2D backbone on 3-slice windows -> tokens tagged with series slot and slice position ->
    small transformer over all windows of the study -> one attention pooling per finding."""

    def __init__(
        self,
        backbone="convnext_tiny",
        pretrained=True,
        dim=384,
        depth=2,
        drop=0.2,
        drop_path=0.1,
        n_views=0,
        spatial=False,
    ):
        super().__init__()
        kw = {"drop_path_rate": drop_path} if drop_path else {}
        self.enc = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, **kw
        )
        self.mean = tuple(self.enc.pretrained_cfg.get("mean", MEAN))
        self.std = tuple(self.enc.pretrained_cfg.get("std", STD))
        self.proj = nn.Sequential(
            nn.Dropout(drop), nn.Linear(self.enc.num_features, dim), nn.LayerNorm(dim)
        )
        self.slot_emb = nn.Embedding(N_SLOTS, dim)
        self.view_emb = nn.Embedding(n_views, dim) if n_views else None
        self.pos_emb = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(
            dim, 8, dim * 2, dropout=0.1, batch_first=True, norm_first=True
        )
        self.tf = (
            nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
            if depth
            else None
        )
        self.att = nn.Sequential(
            nn.Linear(dim, 256),
            nn.Tanh(),
            nn.Dropout(drop),
            nn.Linear(256, len(LABELS)),
        )
        self.cls_w = nn.Parameter(torch.randn(len(LABELS), dim) * 0.02)
        self.cls_b = nn.Parameter(torch.zeros(len(LABELS)))
        self.spatial = spatial
        if spatial:
            self.sp_key = nn.Linear(self.enc.num_features, 128)
            self.sp_pos = nn.Linear(16, 128, bias=False)
            nn.init.zeros_(self.sp_pos.weight)
            self.sp_q = nn.Parameter(torch.zeros(len(LABELS), 128))

    def spatial_pool(self, g):
        hh, ww = g.shape[-2:]
        f = g.flatten(2).transpose(1, 2)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, hh, device=g.device),
            torch.linspace(-1, 1, ww, device=g.device),
            indexing="ij",
        )
        fr = torch.tensor([1.0, 2.0, 4.0, 8.0], device=g.device) * (math.pi / 2)
        yy, xx = yy.reshape(-1, 1) * fr, xx.reshape(-1, 1) * fr
        pe = torch.cat([yy.sin(), yy.cos(), xx.sin(), xx.cos()], 1)
        key = torch.tanh(
            self.sp_key(F.layer_norm(f.float(), (f.shape[-1],)).to(f.dtype))
            + self.sp_pos(pe.to(f.dtype))[None]
        )
        att = (
            torch.einsum("npk,ck->npc", key, self.sp_q.to(key.dtype))
            .float()
            .softmax(1)
            .to(f.dtype)
        )
        out = torch.einsum("npc,npd->ncd", att, f)
        norm = getattr(getattr(self.enc, "head", None), "norm", None)
        if isinstance(norm, nn.LayerNorm):
            out = F.layer_norm(
                out.float(), (out.shape[-1],), norm.weight, norm.bias, norm.eps
            ).to(f.dtype)
        return out

    def head(self, feat, mask, pos, vid=None):
        """feat [B, S, K, D], mask [B, S] bool, pos [B, S, K] in 0..1 -> logits [B, 12]."""
        b, s, k = feat.shape[:3]
        if feat.dim() == 5:
            return self.head_per_class(feat, mask, pos, vid)
        slot = (
            (torch.arange(s, device=feat.device) % N_SLOTS)
            .view(1, s, 1)
            .expand(b, s, k)
        )
        h = (
            self.proj(feat)
            + self.slot_emb(slot)
            + self.pos_emb(pos.unsqueeze(-1).to(feat.dtype))
        )
        if self.view_emb is not None and vid is not None:
            h = h + self.view_emb(vid).unsqueeze(2)
        h = h.view(b, s * k, -1)
        pad = ~mask.unsqueeze(-1).expand(b, s, k).reshape(b, s * k)
        pad = pad & ~pad.all(
            1, keepdim=True
        )  # a study with no series at all attends to everything
        if self.tf is not None:
            h = self.tf(h, src_key_padding_mask=pad)
        a = (
            self.att(h).float().masked_fill(pad.unsqueeze(-1), -1e4).softmax(1)
        )  # [B, T, C]
        pooled = torch.einsum("btc,btd->bcd", a.to(h.dtype), h)
        return (pooled * self.cls_w).sum(-1) + self.cls_b

    def head_per_class(self, feat, mask, pos, vid=None):
        b, s, k, c, _ = feat.shape
        slot = (
            (torch.arange(s, device=feat.device) % N_SLOTS)
            .view(1, s, 1)
            .expand(b, s, k)
        )
        emb = self.slot_emb(slot) + self.pos_emb(pos.unsqueeze(-1).to(feat.dtype))
        if self.view_emb is not None and vid is not None:
            emb = emb + self.view_emb(vid).unsqueeze(2)
        h = self.proj(feat) + emb.unsqueeze(3)
        h = h.view(b, s * k, c, -1).transpose(1, 2).reshape(b * c, s * k, -1)
        pad = ~mask.unsqueeze(-1).expand(b, s, k).reshape(b, s * k)
        pad = pad & ~pad.all(1, keepdim=True)
        if self.tf is not None:
            h = self.tf(h, src_key_padding_mask=pad.repeat_interleave(c, 0))
        h = h.view(b, c, s * k, -1)
        a = torch.diagonal(self.att(h), dim1=1, dim2=3)
        a = a.float().masked_fill(pad.unsqueeze(-1), -1e4).softmax(1)
        pooled = torch.einsum("btc,bctd->bcd", a.to(h.dtype), h)
        return (pooled * self.cls_w).sum(-1) + self.cls_b

    def forward(
        self, x, mask, pos, res=256, train=False, ghost=False, centre=None, vid=None
    ):
        """x uint8 [B, S, K, 3, H, W]."""
        b, s, k = x.shape[:3]
        if centre is not None and centre.dim() == 4:
            p = centre.shape[1]
            x = (
                x.unsqueeze(1)
                .expand(b, p, *x.shape[1:])
                .reshape(b, p * s, *x.shape[2:])
            )
            mask = (mask.unsqueeze(1) & (vid >= 0)).reshape(b, p * s)
            pos = pos.unsqueeze(1).expand(b, p, s, k).reshape(b, p * s, k)
            centre, vid, s = (
                centre.reshape(b, p * s, 2),
                vid.clamp_min(0).reshape(b, p * s),
                p * s,
            )
        keep = mask.reshape(-1)  # run the backbone only on series that exist
        xv = x.reshape(b * s, k, *x.shape[3:])[keep].flatten(0, 1)
        cv = centre.reshape(b * s, 2)[keep] if centre is not None else None
        fov = getattr(self, "fov", 0.92)
        if (
            vid is not None
            and self.view_emb is not None
            and self.view_emb.num_embeddings > N_VIEWS
        ):
            is_full = vid.reshape(b * s)[keep] == FULL_VIEW
            fov = torch.where(
                is_full,
                torch.full_like(is_full, FULL_FOV, dtype=torch.float32),
                torch.full_like(is_full, fov, dtype=torch.float32),
            )
        v = make_views(
            xv,
            res,
            train,
            k,
            fov=fov,
            ghost=ghost,
            centre=cv,
            mean=self.mean,
            std=self.std,
        ).contiguous(memory_format=torch.channels_last)
        f = (
            self.spatial_pool(self.enc.forward_features(v))
            if self.spatial
            else self.enc(v)
        )
        feat = f.new_zeros(b * s, k, *f.shape[1:])
        feat[keep] = f.view(-1, k, *f.shape[1:])
        return self.head(feat.view(b, s, k, *f.shape[1:]), mask, pos, vid)
