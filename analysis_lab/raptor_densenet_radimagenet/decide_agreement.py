import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


DATA = Path(__file__).resolve().parent.parent / "data"
OOF_DIR = DATA / "oof"
RUN_DIR = DATA / "raptor_densenet_runs"
CANDIDATES = {
    "aug_agree": RUN_DIR / "raptor_oof_fold0_e64_reg_swa_fold0.npz",
    "aug": RUN_DIR / "raptor_oof_fold0_e64_aug_swa_fold0.npz",
    "agree": RUN_DIR / "raptor_oof_fold0_e64_agree_swa_fold0.npz",
    "plain": RUN_DIR / "raptor_oof_fold0_e64_swa_fold0.npz",
}
FALLBACK = "aug_agree"
TIE = 0.0002
RAD_WEIGHTS = (0.2, 0.3, 0.4, 0.5, 0.6)
NEW_WEIGHTS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)
N_BOOTSTRAP = 800


def main(out_path):
    dino = np.load(OOF_DIR / "dino_oof.npz")
    targets = [str(t) for t in dino["targets"]]
    ids_all = pd.Index(dino["ids"].astype(str))
    read = lambda p: pd.read_csv(
        p, index_col="StudyInstanceUID", dtype={"StudyInstanceUID": str}
    )[targets]
    preds = {
        "dino": pd.DataFrame(dino["pred"], index=ids_all, columns=targets),
        "e10": read(OOF_DIR / "rad_e10_reference_oof.csv"),
        "e13": read(OOF_DIR / "rad_e13_oof.csv"),
    }
    available = []
    for name, path in CANDIDATES.items():
        if path.exists():
            z = np.load(path)
            preds[name] = pd.DataFrame(z["pred"], index=z["ids"].astype(str), columns=targets)
            available.append(name)
        else:
            print(f"missing {path.name}: {name} left out")
    if not available:
        Path(out_path).write_text(json.dumps({"choice": FALLBACK, "reason": "no candidates found"}))
        print(f"DECISION: {FALLBACK} (fallback, no candidates found)")
        return

    ids = preds[available[0]].index
    labels = pd.DataFrame(dino["y_derived"], index=ids_all, columns=targets).loc[ids]
    ranks = {k: v.loc[ids].rank(pct=True) for k, v in preds.items()}
    y = labels.to_numpy()

    def auc(frame, rows=None):
        p = frame.to_numpy()
        if rows is not None:
            return np.mean([roc_auc_score(y[rows, j], p[rows, j]) for j in range(len(targets))])
        return np.mean([roc_auc_score(y[:, j], p[:, j]) for j in range(len(targets))])

    rad = (ranks["e10"] + ranks["e13"]) / 2
    base = max(auc((1 - w) * ranks["dino"] + w * rad) for w in RAD_WEIGHTS)
    blended, summary = {}, {}
    for k in available:
        options = {
            (w, v): (1 - w - v) * ranks["dino"] + w * rad + v * ranks[k]
            for w in RAD_WEIGHTS
            for v in NEW_WEIGHTS
            if w + v < 1
        }
        scores = {key: auc(frame) for key, frame in options.items()}
        best = max(scores, key=scores.get)
        blended[k] = options[best]
        fixed = 0.5 * ranks["dino"] + 0.5 * (ranks["e10"] + ranks["e13"] + ranks[k]) / 3
        summary[k] = {
            "alone": auc(preds[k].loc[ids]),
            "blend_gain": scores[best] - base,
            "best_weights_rad_new": list(best),
            "blend_gain_fixed_weights": auc(fixed) - base,
            "corr": {
                m: float(np.mean([preds[k][t].corr(preds[m].loc[ids][t], method="spearman") for t in targets]))
                for m in ("dino", "e10", "e13")
            },
        }

    top = max(summary[k]["blend_gain"] for k in available)
    tied = [k for k in available if top - summary[k]["blend_gain"] <= TIE]
    choice = max(tied, key=lambda k: summary[k]["alone"])
    reason = (
        f"best blend gain {top:+.5f}; within {TIE} of it: {tied}; "
        f"most accurate of those alone: {choice}"
    )
    ranked = sorted(available, key=lambda k: -summary[k]["blend_gain"])
    boot = {}
    if len(ranked) > 1:
        rng = np.random.default_rng(0)
        diffs = []
        for _ in range(N_BOOTSTRAP):
            rows = rng.integers(0, len(ids), len(ids))
            try:
                diffs.append(auc(blended[ranked[0]], rows) - auc(blended[ranked[1]], rows))
            except ValueError:
                pass
        diffs = np.array(diffs)
        boot = {
            "comparison": f"{ranked[0]} minus {ranked[1]}",
            "mean": float(diffs.mean()),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "share_positive": float((diffs > 0).mean()),
        }
    decision = {"choice": choice, "reason": reason, "summary": summary, "bootstrap_top_two": boot}
    Path(out_path).write_text(json.dumps(decision, indent=1))
    for k in ranked:
        v = summary[k]
        print(
            f"{k:10s} alone {v['alone']:.5f} | blend gain {v['blend_gain']:+.5f} at weights (rad, new) "
            f"{tuple(v['best_weights_rad_new'])} | fixed-weight gain {v['blend_gain_fixed_weights']:+.5f} "
            f"| corr DINO {v['corr']['dino']:.3f} E10 {v['corr']['e10']:.3f} E13 {v['corr']['e13']:.3f}"
        )
    if boot:
        print(f"bootstrap {boot['comparison']}: mean {boot['mean']:+.5f}, 95% {boot['ci95']}, positive {boot['share_positive']:.0%}")
    print(f"DECISION: {choice} ({reason})")


if __name__ == "__main__":
    main(sys.argv[1])
