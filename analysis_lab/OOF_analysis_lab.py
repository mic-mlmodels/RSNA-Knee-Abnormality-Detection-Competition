from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OOF_DIR = Path(__file__).resolve().parent / "data" / "oof"

# dino ensemble oof folds: known
# resnet rad_e10 oof folds: known
# resnet rad_e13 oof folds: known
# raptor ensemble: didn't leave oof folds
# a5 ensemble: didn't publish gotta ask author
# coat ensemble: gold studies


def load_oof():
    dino = np.load(OOF_DIR / "dino_oof.npz")
    targets = [str(t) for t in dino["targets"]]
    ids = pd.Index(dino["ids"].astype(str), name="StudyInstanceUID")
    rad = pd.read_csv(
        OOF_DIR / "rad_e10_reference_oof.csv", index_col="StudyInstanceUID"
    ).loc[ids]
    e13 = pd.read_csv(OOF_DIR / "rad_e13_oof.csv", index_col="StudyInstanceUID").loc[
        ids
    ]
    if not (rad["fold"] == e13["fold"]).all():
        raise ValueError("Rad OOF fold columns disagree")
    return {
        "targets": targets,
        "labels": pd.DataFrame(dino["y_derived"], index=ids, columns=targets),
        "gold": pd.Series(dino["gold_mask"], index=ids, name="gold"),
        "fold": rad["fold"].rename("fold"),
        "preds": {
            "dino": pd.DataFrame(dino["pred"], index=ids, columns=targets),
            "rad_e10": rad[targets],
            "rad_e13": e13[targets],
        },
    }


def calc_macro_auc(labels, pred):
    return float(np.mean([roc_auc_score(labels[t], pred[t]) for t in labels.columns]))


oof = load_oof()
print(
    f"{len(oof['labels'])} studies, {int(oof['gold'].sum())} gold, folds {oof['fold'].value_counts().sort_index().tolist()}"
)
for name, pred in oof["preds"].items():
    print(f"{name:8} macro AUC {macro_auc(oof['labels'], pred):.4f}")
