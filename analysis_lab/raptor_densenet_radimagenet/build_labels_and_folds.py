import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA = Path(__file__).resolve().parent.parent / "data"
LABEL_DIR = DATA / "labels"
TRAIN_CSV = DATA / "competition" / "train.csv"
FOLD_SOURCE = DATA / "oof" / "rad_e10_reference_oof.csv"
SOFT_LABELS = LABEL_DIR / "labels_llm_soft.parquet"
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
    soft = pd.DataFrame(np.nanmean(cube, axis=0).astype(np.float32), columns=TARGETS)
    soft.insert(0, uid, train[uid])
    soft.to_parquet(SOFT_LABELS, index=False)

    folds = pd.read_csv(FOLD_SOURCE, usecols=[uid, "fold"], dtype={uid: str})
    mapping = {u: int(f) for u, f in zip(folds[uid], folds["fold"])}
    if set(mapping) != set(train[uid]):
        raise ValueError("fold source does not cover the training studies")
    FOLD_FILE.write_text(json.dumps({"folds": mapping}))
    print(f"wrote {SOFT_LABELS} {soft.shape} and {FOLD_FILE} ({len(mapping)} studies)")


if __name__ == "__main__":
    main()
