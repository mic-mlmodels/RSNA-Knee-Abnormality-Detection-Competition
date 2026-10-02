from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


DATA = Path(__file__).resolve().parent.parent / "data"
OOF_DIR = DATA / "oof"
RUN_DIR = DATA / "raptor_densenet_runs"
FROZEN_OOF = DATA / "densenet_run" / "densenet_rad" / "oof.csv"
FOLDS = (0, 1, 2, 3, 4)
RUN_SUFFIX = ""
VARIANT = "_swa"
RAD_WEIGHTS = (0.2, 0.3, 0.4, 0.5, 0.6)
NEW_WEIGHTS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)
N_BOOTSTRAP = 1000
SEED = 2026


def load_base():
    dino = np.load(OOF_DIR / "dino_oof.npz")
    targets = [str(t) for t in dino["targets"]]
    ids = pd.Index(dino["ids"].astype(str), name="StudyInstanceUID")
    read = lambda path: pd.read_csv(
        path, index_col="StudyInstanceUID", dtype={"StudyInstanceUID": str}
    ).loc[ids]
    e10, e13, frozen = (
        read(OOF_DIR / "rad_e10_reference_oof.csv"),
        read(OOF_DIR / "rad_e13_oof.csv"),
        read(FROZEN_OOF),
    )
    return {
        "targets": targets,
        "labels": pd.DataFrame(dino["y_derived"], index=ids, columns=targets),
        "gold": pd.Series(dino["gold_mask"], index=ids),
        "fold": e10["fold"],
        "preds": {
            "dino": pd.DataFrame(dino["pred"], index=ids, columns=targets),
            "e10": e10[targets],
            "e13": e13[targets],
            "frozen": frozen[targets],
        },
    }


def load_new(targets, fold_of):
    parts = []
    for fold in FOLDS:
        z = np.load(RUN_DIR / f"raptor_oof_fold{fold}{RUN_SUFFIX}{VARIANT}_fold{fold}.npz")
        ids = z["ids"].astype(str)
        if not (fold_of.loc[ids] == fold).all():
            raise ValueError(f"fold {fold} file holds studies from another fold")
        parts.append(pd.DataFrame(z["pred"], index=ids, columns=targets))
    new = pd.concat(parts)
    if new.index.duplicated().any():
        raise ValueError("a study appears in more than one fold file")
    return new


def macro_auc(labels, pred):
    return float(np.mean([roc_auc_score(labels[t], pred[t]) for t in labels.columns]))


def per_target_auc(labels, pred):
    return pd.Series({t: roc_auc_score(labels[t], pred[t]) for t in labels.columns})


def blend(ranks, rad_members, w_rad, w_new):
    rad = sum(ranks[m] for m in rad_members) / len(rad_members)
    return (1 - w_rad - w_new) * ranks["dino"] + w_rad * rad + w_new * ranks["new"]


def nested_blend(ranks, labels, fold, rad_members, new_weights):
    out = pd.DataFrame(index=labels.index, columns=labels.columns, dtype=float)
    chosen = []
    for f in FOLDS:
        inner, outer = fold != f, fold == f
        scores = {
            (w, v): macro_auc(
                labels[inner], blend(ranks, rad_members, w, v)[inner]
            )
            for w, v in product(RAD_WEIGHTS, new_weights)
        }
        w, v = max(scores, key=scores.get)
        out[outer] = blend(ranks, rad_members, w, v)[outer]
        chosen.append((w, v))
    return out, chosen


def bootstrap_gain(labels, with_new, without_new):
    rng = np.random.default_rng(SEED)
    y, a, b = labels.to_numpy(), with_new.to_numpy(), without_new.to_numpy()
    gains = []
    for _ in range(N_BOOTSTRAP):
        i = rng.integers(0, len(y), len(y))
        if any(len(np.unique(y[i, j])) < 2 for j in range(y.shape[1])):
            continue
        gains.append(
            np.mean(
                [
                    roc_auc_score(y[i, j], a[i, j]) - roc_auc_score(y[i, j], b[i, j])
                    for j in range(y.shape[1])
                ]
            )
        )
    gains = np.array(gains)
    return gains.mean(), np.percentile(gains, [2.5, 97.5]), (gains > 0).mean()


def report(name, labels, with_new, without_new, chosen):
    base, new = macro_auc(labels, without_new), macro_auc(labels, with_new)
    mean, (lo, hi), share = bootstrap_gain(labels, with_new, without_new)
    print(f"\n{name}")
    print(f"  without {base:.5f} | with {new:.5f} | gain {new - base:+.5f}")
    print(
        f"  bootstrap gain {mean:+.5f}, 95% interval [{lo:+.5f}, {hi:+.5f}], positive in {share:.0%}"
    )
    print(f"  weights chosen per fold (rad, new): {chosen}")
    delta = per_target_auc(labels, with_new) - per_target_auc(labels, without_new)
    print("  per finding: " + ", ".join(f"{t} {d:+.4f}" for t, d in delta.items()))


def main():
    base = load_base()
    targets = base["targets"]
    new = load_new(targets, base["fold"])
    ids = new.index
    labels, fold = base["labels"].loc[ids], base["fold"].loc[ids]
    preds = {name: frame.loc[ids] for name, frame in base["preds"].items()}
    preds["new"] = new
    print(
        f"{len(ids)} studies scored (gold in set: {int(base['gold'].loc[ids].sum())}), "
        f"per fold {fold.value_counts().sort_index().tolist()}"
    )

    print("\nsingle models, macro AUC vs y_derived")
    for name, pred in preds.items():
        per_fold = [macro_auc(labels[fold == f], pred[fold == f]) for f in FOLDS]
        print(
            f"  {name:7} {macro_auc(labels, pred):.5f} | per fold "
            + " ".join(f"{s:.4f}" for s in per_fold)
        )

    ranks = {name: pred.rank(pct=True) for name, pred in preds.items()}
    print("\nmean rank correlation with the new model")
    for name in ("dino", "e10", "e13", "frozen"):
        rho = np.mean(
            [preds["new"][t].corr(preds[name][t], method="spearman") for t in targets]
        )
        print(f"  {name:7} {rho:.3f}")

    without, chosen_without = nested_blend(ranks, labels, fold, ["e10", "e13"], (0.0,))

    in_arm, chosen = nested_blend(ranks, labels, fold, ["e10", "e13", "new"], (0.0,))
    report(
        "test A: new model inside the Rad arm (the test used for the frozen DenseNet)",
        labels,
        in_arm,
        without,
        chosen,
    )
    frozen_arm, chosen = nested_blend(
        ranks, labels, fold, ["e10", "e13", "frozen"], (0.0,)
    )
    report(
        "reference: frozen DenseNet inside the Rad arm", labels, frozen_arm, without, chosen
    )
    own, chosen = nested_blend(ranks, labels, fold, ["e10", "e13"], NEW_WEIGHTS)
    report("test B: new model with its own weight", labels, own, without, chosen)
    print(f"\nbaseline weights chosen per fold (rad, new): {chosen_without}")


if __name__ == "__main__":
    main()
