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
    "agreement": RUN_DIR / "raptor_oof_fold0_e64_reg_swa_fold0.npz",
    "no_agreement": RUN_DIR / "raptor_oof_fold0_e64_aug_swa_fold0.npz",
}
TIE = 0.0002
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
    for name, path in CANDIDATES.items():
        if not path.exists():
            decision = {"choice": "agreement", "reason": f"missing {path.name}"}
            Path(out_path).write_text(json.dumps(decision, indent=1))
            print(f"DECISION: agreement (fallback, {path.name} missing)")
            return
        z = np.load(path)
        preds[name] = pd.DataFrame(z["pred"], index=z["ids"].astype(str), columns=targets)

    ids = preds["agreement"].index
    labels = pd.DataFrame(dino["y_derived"], index=ids_all, columns=targets).loc[ids]
    ranks = {k: v.loc[ids].rank(pct=True) for k, v in preds.items()}
    y = labels.to_numpy()

    def auc(frame, rows=None):
        p = frame.to_numpy()
        if rows is not None:
            return np.mean([roc_auc_score(y[rows, j], p[rows, j]) for j in range(len(targets))])
        return np.mean([roc_auc_score(y[:, j], p[:, j]) for j in range(len(targets))])

    without = 0.6 * ranks["dino"] + 0.4 * (ranks["e10"] + ranks["e13"]) / 2
    blended = {
        k: 0.5 * ranks["dino"] + 0.5 * (ranks["e10"] + ranks["e13"] + ranks[k]) / 3
        for k in CANDIDATES
    }
    base = auc(without)
    summary = {}
    for k in CANDIDATES:
        summary[k] = {
            "alone": auc(preds[k].loc[ids]),
            "blend_gain": auc(blended[k]) - base,
            "corr_dino": float(
                np.mean([preds[k][t].corr(preds["dino"].loc[ids][t], method="spearman") for t in targets])
            ),
        }
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(N_BOOTSTRAP):
        rows = rng.integers(0, len(ids), len(ids))
        try:
            diffs.append(auc(blended["no_agreement"], rows) - auc(blended["agreement"], rows))
        except ValueError:
            pass
    diffs = np.array(diffs)
    gap = summary["no_agreement"]["blend_gain"] - summary["agreement"]["blend_gain"]
    if abs(gap) > TIE:
        choice = "no_agreement" if gap > 0 else "agreement"
        reason = f"blend gain differs by {gap:+.5f}"
    else:
        alone_gap = summary["no_agreement"]["alone"] - summary["agreement"]["alone"]
        choice = "no_agreement" if alone_gap > 0 else "agreement"
        reason = f"blend gains tied ({gap:+.5f}); single-model AUC differs by {alone_gap:+.5f}"
    decision = {
        "choice": choice,
        "reason": reason,
        "summary": summary,
        "bootstrap_no_minus_with": {
            "mean": float(diffs.mean()),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "share_positive": float((diffs > 0).mean()),
        },
    }
    Path(out_path).write_text(json.dumps(decision, indent=1))
    for k, v in summary.items():
        print(
            f"{k:13s} alone {v['alone']:.5f} | blend gain {v['blend_gain']:+.5f} | corr DINO {v['corr_dino']:.3f}"
        )
    print(f"bootstrap (no_agreement - agreement, blend): {decision['bootstrap_no_minus_with']}")
    print(f"DECISION: {choice} ({reason})")


if __name__ == "__main__":
    main(sys.argv[1])
