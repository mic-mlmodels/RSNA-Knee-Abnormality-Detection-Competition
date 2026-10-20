import sys
import time

import numpy as np
import pandas as pd
import torch

from knee import LABELS, N_SLOTS, KneeNet, build_study_table, window_centres

D = "../data/"
RUN = D + "reader_runs/" + (sys.argv[2] if len(sys.argv) > 2 else "lab_all3") + "/"
CACHE = D + "reader_work/cache384_train"
SAG = [0, 3]


def fill_table(table):
    out = {}
    for st, slots in table.items():
        slots = [list(c) for c in slots]
        for p in range(3):
            if not slots[p] and slots[p + 3]:
                slots[p] = list(slots[p + 3])
            elif not slots[p + 3] and slots[p]:
                slots[p + 3] = list(slots[p])
        out[st] = slots
    return out


class CacheStudies(torch.utils.data.Dataset):
    def __init__(self, studies, table, k, offset):
        self.studies, self.table, self.k, self.offset = list(studies), table, k, offset

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        slots = self.table.get(self.studies[i], [[] for _ in range(N_SLOTS)])
        x = np.zeros((N_SLOTS, self.k, 3, 384, 384), np.uint8)
        mask, pos = np.zeros(N_SLOTS, bool), np.zeros((N_SLOTS, self.k), np.float32)
        for s, cands in enumerate(slots):
            if not cands:
                continue
            vol = np.load(f"{CACHE}/{cands[0]}.npy", mmap_mode="r")
            n = len(vol)
            if n < 3:
                continue
            c = window_centres(n, self.k, False)
            if self.offset:
                c = np.clip(c + self.offset, 1, max(1, n - 2))
            x[s] = np.stack([vol[np.clip([j - 1, j, j + 1], 0, n - 1)] for j in c])
            mask[s], pos[s] = True, c / max(n - 1, 1)
        return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(pos)


def anatomical_mirror(x, pos):
    xm, pm = x.flip(-1), pos.clone()
    xm[:, SAG] = x[:, SAG].flip(3)
    pm[:, SAG] = 1 - pos[:, SAG]
    return xm, pm


@torch.no_grad()
def main():
    fold = int(sys.argv[1])
    lab = pd.read_csv(D + "reader_work/labels_v0.csv", index_col=0)
    meta = pd.read_csv(D + "reader_work/cache384_train_meta.csv")
    plain = build_study_table(pd.read_csv(D + "competition/train_series.csv"), dict(zip(meta.series, meta.n.fillna(0))))
    tables = {"plain": plain, "fill": fill_table(plain)}
    studies = list(lab[(lab.fold == fold) | (lab.is_gold == 1)].index)
    ck = torch.load(f"{RUN}fold{fold}_e18.pt", map_location="cpu", weights_only=False)
    model = KneeNet(ck["args"]["backbone"], pretrained=False)
    model.load_state_dict(ck["model"])
    model = model.cuda().to(memory_format=torch.channels_last).eval()
    passes = [("plain", 16, off, [(r, fl) for r in (256, 320) for fl in ("none", "hflip", "anat")]) for off in (0, 1)]
    keys, preds, t0 = [], [], time.time()
    for tname, k, off, variants in passes:
        dl = torch.utils.data.DataLoader(CacheStudies(studies, tables[tname], k, off), batch_size=2, num_workers=4, prefetch_factor=2)
        outs = {v: [] for v in variants}
        for x, mask, pos in dl:
            x, mask, pos = x.cuda(non_blocking=True), mask.cuda(), pos.cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                for res, flip in variants:
                    if flip == "none":
                        p = model(x, mask, pos, res=res)
                    elif flip == "hflip":
                        p = model(x.flip(-1), mask, pos, res=res)
                    else:
                        xm, pm = anatomical_mirror(x, pos)
                        p = model(xm, mask, pm, res=res)
                    outs[(res, flip)].append(p.float().sigmoid().cpu())
        for (res, flip), v in outs.items():
            keys.append(f"{tname}|k{k}|off{off}|r{res}|{flip}")
            preds.append(torch.cat(v).numpy())
        print(f"fold {fold}: pass {tname} k{k} offset {off} done ({time.time() - t0:.0f}s)", flush=True)
    np.savez_compressed(D + f"reader_runs/tta_audit3_" + (sys.argv[2] if len(sys.argv) > 2 else "lab_all3") + f"_fold{fold}.npz", keys=np.array(keys), preds=np.stack(preds).astype(np.float32), ids=np.array(studies))
    print(f"fold {fold}: saved {len(keys)} variants for {len(studies)} studies", flush=True)


if __name__ == "__main__":
    main()
