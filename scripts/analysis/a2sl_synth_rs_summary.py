"""Summarise a half-synthetic run root: censored log score and day error (MAE) of the tracked fruit, by start, on the
synthetic test fruit (``predictions_test``) and on their real visits (``predictions_test_real``).

A test prediction of fruit ``avo01@k`` is avo01 seen from its k-th visit on (start day 1 + 2k), time re-zeroed;
plain ``avo01`` is the usual day-1 start. Per seed: predictions pooled over the 4 folds, mean over every forecast
(each visible prefix); cells are mean ± std over seeds. ``all starts`` weights every start equally.

python scripts/analysis/a2sl_synth_rs_summary.py [--root outputs/a2sl_synth_rs]   -> <root>/summary.md (+ printed)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ripen.eval.metrics import censored_log_score, compute_metrics

TRACKED = ("avo01", "avo02", "avo03", "avo11")
# reference row in every table: temporal matching on the date alone = "how many days fruit usually have left"
REF_ROOT, REF_RUN = Path("outputs/a2sl_synth_rs"), "matching_time_synth-linear_start-random"
REF_NAME = "**reference: no-photo forecast**"


def split_id(i: str) -> tuple[str, int]:
    base, _, k = str(i).partition("@")
    return base, int(k) if k else 0


def pooled(run_dir: Path, split: str = "test") -> dict | None:
    parts = [dict(np.load(f, allow_pickle=True)) for f in sorted(run_dir.glob(f"fold*/predictions_{split}.npz"))]
    if len(parts) < 4:
        return None
    return {k: np.concatenate([p[k] for p in parts]) for k in ("F", "lo", "hi", "item_ids")}


def scores(z: dict) -> dict[int, tuple[float, float]]:
    """start index -> (mean log score, MAE) over the tracked fruit's forecasts."""
    ids = [split_id(i) for i in z["item_ids"]]
    out = {}
    for k in sorted({k for _, k in ids}):
        m = np.array([b in TRACKED and kk == k for b, kk in ids])
        if m.any():
            lg = float(np.mean(censored_log_score(z["F"][m], z["lo"][m], z["hi"][m])))
            out[k] = (lg, compute_metrics(z["F"][m], z["lo"][m], z["hi"][m]).to_dict()["mae"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/a2sl_synth_rs")
    a = ap.parse_args()
    root = Path(a.root)
    md, out = [], {}
    for split, title in (("test_real", "Real visits of the tracked fruit"), ("test", "Synthetic test fruit")):
        t, j = table(root, split)
        if t:
            md += [f"## {title} (predictions_{split})", "", t, ""]
            out[split] = j
    text = "Censored log score (lower is better) of the 4 tracked fruit, by start; mean ± std over seeds. "
    text += "The reference row is the no-photo forecast (matching on the date alone); a model that cannot beat it is "
    text += "not using the photos usefully.\n\n"
    text += "\n".join(md)
    (root / "summary.md").write_text(text)
    (root / "summary.json").write_text(json.dumps(out, indent=1))
    print(text)


def table(root: Path, split: str) -> tuple[str, dict]:
    res: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    dirs = [(rd.name, rd) for sd in sorted(root.glob("seed*")) for rd in sorted(p for p in sd.iterdir() if p.is_dir())]
    dirs += [(REF_NAME, rd) for rd in sorted(REF_ROOT.glob(f"seed*/{REF_RUN}"))]
    for name, rd in dirs:
        z = pooled(rd, split)
        if z is None:
            continue
        sc = scores(z)
        for k, v in sc.items():
            res[name][k].append(v)
        res[name][-1].append((float(np.mean([v[0] for v in sc.values()])), float(np.mean([v[1] for v in sc.values()]))))
    if not res:
        return "", {}
    starts = sorted({k for r in res.values() for k in r if k >= 0})
    hdr = ["run", "seeds", "all starts"] + [f"start day {1 + 2 * k}" for k in starts] + ["MAE (all starts)"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]

    def cell(v, j=0):
        x = np.array([t[j] for t in v])
        return "—" if len(x) == 0 else f"{x.mean():.2f} ± {x.std(ddof=1) if len(x) > 1 else 0:.2f}"

    out = {}
    for run, r in sorted(res.items(), key=lambda kv: (kv[0] != REF_NAME, kv[0])):
        lines.append("| " + " | ".join([run, str(len(r[-1])), cell(r[-1])] + [cell(r.get(k, [])) for k in starts]
                                        + [cell(r[-1], 1)]) + " |")  # fmt: skip
        out[run] = {str(k): [list(t) for t in v] for k, v in r.items()}
    return "\n".join(lines), out


if __name__ == "__main__":
    main()
