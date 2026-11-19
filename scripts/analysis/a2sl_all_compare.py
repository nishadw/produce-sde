"""All-fruit A2SL study (docs/plans/2026-11-19_a2sl-all-fruit.md): every model against the no-photo forecast.

Real photos only, random-start training, one fold per held-out fruit (avo01-12; avo13-15 were first photographed
already spoiled, so they only train). Each held-out fruit is scored from every start day on its real visits. Two
scores per model: all 12 fruit, and the 8 whose spoilage was observed (seen unspoiled, then spoiled); the other 4 were
cut before day 9, so they only say "not spoiled yet" and are easy for any forecast. The paired test against the
no-photo forecast, the "fruit better" count and the day error use the 8 observed fruit. Scoring and the bootstrap are
``a2sl_pop_compare``'s (each start day weighted equally; 95 % bootstrap over fruit).

python scripts/analysis/a2sl_all_compare.py   -> outputs/a2sl_all_compare/{summary.md,summary.json}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a2sl_pop_compare as pc  # noqa: E402

GRID, AML = Path("outputs/a2sl_all"), Path("outputs/a2sl_all_aml")
OUT = Path("outputs/a2sl_all_compare")
ALL = [f"avo{i:02d}" for i in range(1, 13)]
OBSERVED = ["avo01", "avo02", "avo03", "avo08", "avo09", "avo10", "avo11", "avo12"]
N_RUNS = 12 * 5  # folds x seeds
pc.TRACKED, pc.SPLIT = tuple(ALL), "test"
pc.POP = POP = (GRID, "matching_time_synth-real_start-random")
SECTIONS = ("Reference", "Date only (no images)", "Photo", "Photo-only model taught by a teacher during training")
CURVE_ROWS = [("Latent SDE", "photo"), ("Latent ODE", "photo"), ("Transformer", "photo"),
              ("Latent SDE", "SAMD, teacher sees photo + scanner"),
              ("Transformer", "CAML-KD, teacher sees photo only")]  # fmt: skip


def complete(spec: tuple) -> bool:
    return len(list(spec[0].glob(f"seed*/{spec[1]}/fold*/predictions_test.npz"))) >= N_RUNS


def rows() -> list[dict]:
    def g(m, inp):
        return (GRID, f"{m}_{inp}_synth-real_start-random")

    out = [{"section": SECTIONS[0], "model": "No-photo forecast", "sees": "date only", "spec": POP}]
    out += [{"section": SECTIONS[1], "model": lab, "sees": "date only", "spec": g(m, "time")}
            for m, lab in pc.MODELS.items() if m != "matching"]  # fmt: skip
    out += [{"section": SECTIONS[2], "model": lab, "sees": "photo", "spec": g(m, "photo")}
            for m, lab in pc.MODELS.items()]  # fmt: skip
    out += [{"section": SECTIONS[3], "model": pc.MODELS[m], "sees": d, "spec": (AML, f"{m}_{me}_viewall")}
            for m in pc.TAUGHT for me, d in pc.METHODS.items()]  # fmt: skip
    return [r for r in out if complete(r["spec"])]


def only(pf: dict, fruit: list[str]) -> dict:
    return {k: v for k, v in pf.items() if v[0] in fruit}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pop = pc.per_forecast(POP)[0]
    ref_all, ref_obs = pc.summary(pop)[0], pc.summary(only(pop, OBSERVED))[0]
    n = len(OBSERVED)
    out: dict = {"ref_all": ref_all, "ref_obs": ref_obs, "rows": [], "curve": {}}
    md = ["# All 15 avocados, real photos only: every model against the no-photo forecast", "",
          "How to read: log score on the held-out fruit's real photos, from every start day (lower is better); "
          "random-start training. *All 12* = every fruit with an unspoiled photo; *8 observed* = the fruit seen "
          "unspoiled and then spoiled (the fair comparison: the other 4 were cut early and only say 'not spoiled "
          f"yet'). *vs no-photo* = 8-observed score minus the no-photo forecast ({ref_obs:.2f}); 'better' / 'worse' "
          "only when the 95 % bootstrap over fruit excludes 0. *Fruit better* = of the 8, how many the model beats "
          "the no-photo forecast on. *Day error* = days between the forecast's middle and the observed spoilage day. "
          "*Blend* = mixed with the no-photo forecast, the mix chosen on validation fruit (w = no-photo share).",
          "", "| Model | Sees | All 12 fruit | 8 observed | vs no-photo | Fruit better | Day error | Blend (w) |",
          "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    section = None
    for r in rows():
        if r["section"] != section:
            section = r["section"]
            md.append(f"| **{section}** | | | | | | | |")
        pf = pc.per_forecast(r["spec"])[0]
        lg_all = pc.summary(pf)[0]
        po = only(pf, OBSERVED)
        lg_obs, mae = pc.summary(po)
        rec = {k: r[k] for k in ("section", "model", "sees")} | {"all": lg_all, "obs": lg_obs, "mae": mae}
        vs = wins = blend = "—"
        if r["spec"] != POP:
            c = pc.compare(po, only(pop, OBSERVED))
            rec["vs"], rec["vs_all"] = c, pc.compare(pf, pop)
            vs, wins = pc.verdict(c), f"{c['wins']}/{n}"
            if r["model"] != "Image matching" and r["sees"] != "date only":
                bpf, ws = pc.per_forecast(r["spec"], blend=True)
                rec["blend"], rec["w"] = pc.summary(only(bpf, OBSERVED))[0], ws
                blend = f"{rec['blend']:.2f} ({np.median(ws):.2f})"
            if (r["model"], r["sees"]) in CURVE_ROWS:
                out["curve"][f"{r['model']}, {r['sees']}"] = {
                    f"{w:.1f}": pc.summary(only(pc.per_forecast(r["spec"], fixed_w=float(w))[0], OBSERVED))[0]
                    for w in pc.CURVE
                }
        md.append(f"| {r['model']} | {r['sees']} | {lg_all:.2f} | {lg_obs:.2f} | {vs} | {wins} | {mae:.1f} d | "
                  f"{blend} |")  # fmt: skip
        out["rows"].append(rec)
    md += ["", "## Diagnostic: blend scored on the 8 observed test fruit at every w (not a valid score)", "",
           "| Model | " + " | ".join(f"w={w:.1f}" for w in pc.CURVE) + " |",
           "|---|" + "---|" * len(pc.CURVE)]  # fmt: skip
    for lab, c in out["curve"].items():
        md.append(f"| {lab} | " + " | ".join(f"{v:.2f}" for v in c.values()) + " |")
    text = "\n".join(md) + "\n"
    (OUT / "summary.md").write_text(text)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    print(text)


if __name__ == "__main__":
    main()
