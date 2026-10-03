import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA = Path(__file__).resolve().parent.parent / "data"
LABEL_DIR = DATA / "labels"
TRAIN_CSV = DATA / "competition" / "train.csv"
FOLD_SOURCE = DATA / "oof" / "rad_e10_reference_oof.csv"
SOFT_LABELS = LABEL_DIR / "labels_llm_soft.parquet"
AGREEMENT_WEIGHTS = LABEL_DIR / "labels_llm_weights.parquet"
FOLD_FILE = LABEL_DIR / "folds.json"

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


def main():
    uid = "StudyInstanceUID"
    train = pd.read_csv(TRAIN_CSV, dtype={uid: str})
    sources = [
        pd.read_csv(LABEL_DIR / "report_labels_v2.csv", dtype={uid: str}),
        pd.read_csv(LABEL_DIR / "llm_labels_v2.csv", dtype={uid: str}),
        pd.read_csv(LABEL_DIR / "labels_llm_gpt56sol.csv", dtype={uid: str}),
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
    y = np.nanmean(cube, axis=0)
    soft = pd.DataFrame(y.astype(np.float32), columns=TARGETS)
    soft.insert(0, uid, train[uid])
    soft.to_parquet(SOFT_LABELS, index=False)

    disagreement = np.nanmean(np.abs(cube - y[None]), axis=0)
    agreement = np.clip(1.0 - 2.0 * disagreement, 0, 1)
    certainty = np.clip(2.0 * np.abs(y - 0.5), 0, 1)
    w = 0.15 + 0.85 * (0.65 * agreement + 0.35 * certainty)
    weights = pd.DataFrame(w.astype(np.float32), columns=TARGETS)
    weights.insert(0, uid, train[uid])
    weights.to_parquet(AGREEMENT_WEIGHTS, index=False)

    folds = pd.read_csv(FOLD_SOURCE, usecols=[uid, "fold"], dtype={uid: str})
    mapping = {u: int(f) for u, f in zip(folds[uid], folds["fold"])}
    if set(mapping) != set(train[uid]):
        raise ValueError("fold source does not cover the training studies")
    FOLD_FILE.write_text(json.dumps({"folds": mapping}))
    print(
        f"wrote {SOFT_LABELS} {soft.shape}, {AGREEMENT_WEIGHTS} {weights.shape} "
        f"and {FOLD_FILE} ({len(mapping)} studies)"
    )


if __name__ == "__main__":
    main()
