"""Mango views / time series / AML: outputs/mango_views/seed*/<run>/ -> <root>/summary.md.

Run names (``scripts/run/mango_cv.py``): ``<model>_<setting>[tag][_v<view>][_side<k>][_hist]``; fused runs
(``late_fusion.py mango``) end in ``-mean`` / ``-geo`` / ``-trust``. Scores as ``mango_summary.py``; the paired
bootstrap (population Brier per fruit, fruit resampled) compares each row with the section's reference.

    python scripts/analysis/mango_views_summary.py [--root outputs/mango_views_recal]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mango_summary as ms  # noqa: E402

from ripen.train.aml import METHODS as AML_METHODS  # noqa: E402

COLS = [
    ("cutday", "pop_brier", "population Brier"),
    ("cutday", "status_brier", "spoiled-now Brier"),
    ("cutday", "spearman_within_day", "within-day Spearman"),
    ("prefix", "log_score", "earlier-scan log score"),
]
VIEW_LABEL = {f"sep{s}": f"antennas {s} apart" for s in range(1, 6)} | {"img": "image reconstruction",
                                                                       "concat": "all six side by side"}  # fmt: skip
SETTING = {"rgb": "photo", "mw": "microwave", "both": "photo + microwave", "time": "elapsed time only"}


CLOCK = {
    "age_clock": "known age + batch curve",
    "rgb_clock": "photo clock",
    "mw_clock": "microwave clock",
    "both_clock": "photo + microwave clock",
}
FUSE = {"mean": "average", "geo": "geometric mean", "trust": "confidence-weighted"}


def describe(run: str) -> str:
    if run in CLOCK:
        return CLOCK[run]
    aml = "|".join(sorted(AML_METHODS, key=len, reverse=True))
    m = re.match(rf"(latent_sde|latent_ode|neural_cde|gru_d|[a-z_]+?)_(rgb|mw|both|time|{aml})(.*)$", run)
    if not m:
        return run
    model, setting, rest = m.groups()
    fuse = re.search(r"-(mean|geo|trust)$", rest)
    rest = re.sub(r"-(mean|geo|trust)$", "", rest)
    view = re.search(r"_v([a-z0-9+]+)", rest)
    side = re.search(r"_side(\d)", rest)
    tag = re.match(r"(_[a-z0-9]+?)(?=_v|_side|_hist|$)", rest)
    parts = [model.replace("_", " "), SETTING.get(setting, f"AML {setting}")]
    if tag and not tag.group(1).startswith(("_v", "_side", "_hist")):
        parts[-1] += tag.group(1).replace("_", " ")
    if view and view.group(1) == "fuse":
        parts.append(f"six separate models, combined ({FUSE[fuse.group(1)]})")
        fuse = None
    elif view:
        v = view.group(1)
        parts.append(VIEW_LABEL.get(v, v.replace("sep", "").replace("+", ", ") + " apart, pooled" if "+" in v else v))
    if fuse:
        parts.append(f"combined ({FUSE[fuse.group(1)]})")
    if side:
        parts.append(f"photo side {side.group(1)}")
    parts.append("scan history" if "_hist" in rest else "one scan")
    return ", ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/mango_views")
    a = ap.parse_args()
    ms.ROOT = Path(a.root)
    runs = ms.load()
    names = sorted(runs)

    def row(run, ref):
        cells = [ms.ms([mm.get(b, {}).get(k, float("nan")) for mm in runs[run]]) for b, k, _ in COLS]
        cmp = "—" if run == ref or ref not in runs else ms.bootstrap(run, ref)
        return f"| {describe(run)} | " + " | ".join(cells) + f" | {cmp} |"

    def section(title, sel, ref, note=""):
        rs = [r for r in names if sel(r)]
        if not rs:
            return []
        out = [f"## {title}", ""] + ([note, ""] if note else [])
        out += ["| run | " + " | ".join(c[2] for c in COLS) + f" | pop. Brier vs {describe(ref)} |",
                "|---|" + "---|" * (len(COLS) + 1)]  # fmt: skip
        return out + [row(r, ref) for r in sorted(rs, key=lambda r: (r != ref, r))] + [""]

    aml = tuple(AML_METHODS)
    is_aml = lambda r: re.match(rf"[a-z_]+?_({'|'.join(aml)})(_|-|$)", r) is not None and "_both" not in r  # noqa: E731
    L = [f"# Mango: views, time series and AML (`{a.root}`)", "",
         "Cells: mean ± std over seeds; 6 folds by fruit; scored on the cut day without stage labels. Bootstrap: "
         "first minus reference, 95 % interval, `*` = excludes 0 (negative = better).", ""]  # fmt: skip
    L += section("References", lambda r: r.endswith("_clock"), "age_clock")
    L += section("1. Single views (latent SDE, one scan)",
                 lambda r: r.startswith("latent_sde_") and not r.endswith("_hist") and not is_aml(r) and
                 (re.search(r"_v(sep\d|img)$", r) or re.match(r"latent_sde_rgb(_side\d)?$", r)),
                 "latent_sde_rgb")  # fmt: skip
    L += section("2. All views in one model (latent SDE, one scan)",
                 lambda r: r.startswith("latent_sde_") and not r.endswith("_hist") and not is_aml(r) and
                 (re.search(r"_v(concat|sep[0-9+]+\+[0-9sep+]+)", r) or re.search(r"-(mean|geo|trust)$", r) or
                  r in ("latent_sde_mw", "latent_sde_both")), "latent_sde_rgb")  # fmt: skip
    L += section("3. Time-series models with the scan history",
                 lambda r: r.endswith("_hist") and not is_aml(r) and "-" not in r, "age_clock",
                 "Every instance sees all scans up to now; elapsed time is known, so the reference is the known-age "
                 "batch curve.")  # fmt: skip
    L += section("3b. Same models, one scan", lambda r: not r.endswith("_hist") and not is_aml(r) and
                 "_".join(r.split("_")[:2]) in ("latent_ode", "neural_cde", "gru_d") and "_clock" not in r,
                 "latent_sde_rgb")  # fmt: skip
    pat = rf"_({'|'.join(aml)})[a-z0-9_]*?(?=_v|_side|_hist|$)"
    for base in sorted({re.sub(pat, "_rgb", r) for r in names if is_aml(r)}):
        model = base.split("_rgb")[0]
        suffix = base[len(model) + 4 :]
        ref = f"{model}_rgb{re.sub(r'_v[a-z0-9+]+', '', suffix)}"
        L += section(f"4. AML: photo-only students ({describe(base)})",
                     lambda r, model=model, suffix=suffix, ref=ref: (is_aml(r) and r.startswith(model + "_")
                                                                     and r.endswith(suffix))
                     or r in (ref, f"{model}_both{suffix}"), ref)  # fmt: skip
    text = "\n".join(L) + "\n"
    (ms.ROOT / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
