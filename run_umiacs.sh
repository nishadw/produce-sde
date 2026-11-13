#!/usr/bin/env bash
# One file to run the whole project on UMIACS (or any SLURM cluster, or a workstation with --local).
#
#   bash run_umiacs.sh pipeline         # setup -> data -> all as one chain of sbatch jobs (the usual entry point)
#   bash run_umiacs.sh setup            # sbatch job: venv + editable install + CPU tests (run once)
#   bash run_umiacs.sh data             # sbatch job: download avocado if incomplete, check / prepare every dataset
#   bash run_umiacs.sh all              # submit every experiment with dependencies, then the summary job
#   bash run_umiacs.sh e1|e2|e3|e4|e6|synthetic|summary      # one stage
#   bash run_umiacs.sh all --local      # run everything sequentially in this shell (no sbatch)
#
# setup and data run inside sbatch jobs too (no compute on submission nodes); `--local` runs any stage inline.
# SLURM identifiers: on UMIACS class accounts (nexusclass*) they default to account=class partition=class
# qos=default; elsewhere set RIPEN_ACCOUNT / RIPEN_PARTITION / RIPEN_QOS
# (`sacctmgr show assoc user=$USER format=account,partition,qos` lists them). Everything else is a default you
# can override the same way: RIPEN_ENV (venv path), RIPEN_ROOT (repo), RIPEN_GPU (gres string, default gpu:1),
# RIPEN_PYTHON (interpreter for the venv, >=3.10), RIPEN_AFTER (job ids every submitted job waits for).
# Per-machine values can live in .ripen.env (gitignored), sourced first.
#
# Experiments (see docs/plans/experiment_plan.md):
#   synthetic  fake-fruit recovery at full scale                     outputs/synthetic/
#   e1         every model on the Mendeley avocado set               outputs/runs/<model>_avocado
#   e2         exact filter vs PDE-ELBO vs Monte Carlo likelihoods    outputs/runs/{fpt_filter,latent_sde}_*
#   e3         leave-one-storage-group-out transfer + clock ablation outputs/runs/*_logo_*
#   e4         sampling-frequency and photo-count ablations          outputs/runs/*_every{2,3}, metrics_by_n_obs.json
#   e6         A2SL multimodal avocados (photos + microwave)         outputs/runs/<model>_a2sl
#   summary    outputs/summary.md
#   seeds      independent multi-seed rerun of e1-e4 (avocado only)  outputs/avocado/round1/, summary.md there
#   Round 2 (docs/plans/round2_plan.md):
#   r2         guided-posterior latent SDE (ResNet-18 features), 5 seeds: in-distribution + 3 held-out groups
#              outputs/avocado/round2/
#   nb-sweep   new baselines, seed 1, small config sweep (pick by val CRPS) -> outputs/avocado/baselines_sweep/
#   nb         new baselines x {E1, LOGO x3, every2/3, A2SL}, 5 seeds; NB_ARGS_<model>='...' passes the chosen config
#   r2b        the rest for the guided model: labels-only, MC likelihood, clock ablation, sparse photos, and
#              A2SL (guided latent SDE, GRU-D, FPT filter)   outputs/avocado/round2/
#   recal      post-hoc PIT recalibration (val -> test) of every saved run  metrics_test_recal.json in each run
#   Mango (docs/experiments/2026-11-06_mango-photo-vs-microwave/report.md):
#   mango-data ingest data/mango (manifests + microwave cache) and photo features
#   mango      every model x {rgb, mw, both, samd}, 6-fold CV x 3 seeds, CPU   outputs/mango/, summary.md there
#   a2sl-views which of the 6 microwave views predicts best (time series + snapshot)   outputs/a2sl_views/
#   a2sl-aml   AML / CAML / SAMD: photo + microwave teacher -> photo-only students         outputs/a2sl_aml/
#   mango-views mango per-view / time-series / AML runs, one per line of RIPEN_MANGO_RUNS   outputs/mango_views/
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck disable=SC1091
[[ -f .ripen.env ]] && source .ripen.env
if [[ "$(hostname)" == nexusclass* ]]; then
  export RIPEN_ACCOUNT="${RIPEN_ACCOUNT:-class}" RIPEN_PARTITION="${RIPEN_PARTITION:-class}" RIPEN_QOS="${RIPEN_QOS:-default}"
fi
export RIPEN_ROOT="${RIPEN_ROOT:-$PWD}"
export RIPEN_ENV="${RIPEN_ENV:-$HOME/envs/ripen}"
export RIPEN_GPU="${RIPEN_GPU:-gpu:1}"
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
STAGE="${1:-all}"; LOCAL=0; [[ "${2:-}" == "--local" ]] && LOCAL=1
PY="${RIPEN_ENV}/bin/python"
read -r -a AFTER <<< "${RIPEN_AFTER:-}"
declare -A JOBS
MODELS=(stage_linear direct_reg discrete_surv gru_gap gru_d ode_rnn latent_ode latent_sde fpt_filter)
mkdir -p outputs/logs outputs/runs

die() { echo "error: $*" >&2; exit 1; }

# submit NAME "command" [dependency-job-ids...]  -> prints the job id (or runs inline with --local)
submit() {
  local name="$1" cmd="$2"; shift 2
  if [[ $LOCAL -eq 1 ]]; then echo "### $name"; bash -c "$cmd"; echo "local"; return; fi
  submit_raw "$name" "source '$RIPEN_ENV/bin/activate' && cd '$RIPEN_ROOT' && $cmd" 08:00:00 "$@"
}

# submit_raw NAME "command" TIME [dependency-job-ids...]  -> sbatch without activating the venv; prints the job id
submit_raw() {
  local name="$1" cmd="$2" time="$3"; shift 3
  [[ "${RIPEN_ACCOUNT:-CHANGE_ME}" == "CHANGE_ME" || -z "${RIPEN_ACCOUNT:-}" ]] && die "set RIPEN_ACCOUNT / RIPEN_PARTITION / RIPEN_QOS first"
  local deps=("${AFTER[@]}" "$@") dep=""; [[ ${#deps[@]} -gt 0 ]] && dep="--dependency=afterok:$(IFS=:; echo "${deps[*]}")"
  local gres="--gres=$RIPEN_GPU"; [[ "$RIPEN_GPU" == none ]] && gres=""  # RIPEN_GPU=none: CPU-only job
  sbatch --parsable --job-name="ripen-$name" --output="outputs/logs/%x-%j.out" --time="$time" --cpus-per-task="${RIPEN_CPUS:-4}" --mem="${RIPEN_MEM:-16G}" \
    $gres --account="$RIPEN_ACCOUNT" --partition="$RIPEN_PARTITION" --qos="$RIPEN_QOS" $dep \
    --wrap="cd '$RIPEN_ROOT' && $cmd"
}

# setup and data are whole stages; outside --local they resubmit this script as one sbatch job
self_job() { JOBS[$1]=$(submit_raw "$1" "bash run_umiacs.sh $1 --local" "$2" "${@:3}"); }

stage_setup() {
  [[ -d "$RIPEN_ENV" ]] || "${RIPEN_PYTHON:-python3}" -m venv "$RIPEN_ENV"
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -e ".[dev,features]"
  if command -v nvidia-smi >/dev/null 2>&1; then "$PY" -m pip install -q -U "jax[cuda12]"; fi
  JAX_PLATFORMS=cpu "$PY" -m pytest -q -m "not slow"
  JAX_PLATFORMS=cpu "$PY" scripts/run/smoke_all.py
}

stage_data() {
  local n_img=0; [[ -d data/avocado/images ]] && n_img=$(find data/avocado/images -name '*.jpg' | wc -l)
  [[ $n_img -ge 14710 ]] || "$PY" scripts/data/download_avocado.py --workers 8 || die "avocado download incomplete; rerun data"
  "$PY" -c "from ripen.data.avocado import load_avocado; n = len(load_avocado('data/avocado')); print('avocado items', n); assert n == 478"
  [[ -d data/a2sl ]] && "$PY" scripts/data/ingest_a2sl.py || echo "note: data/a2sl not present; e6 will be skipped"
  # the loaders read the feature caches, so extract them before the splits
  [[ -f data/features/avocado_resnet18.npz ]] || "$PY" scripts/data/extract_features.py --data avocado --root data/avocado --out data/features/avocado_resnet18.npz --device "$( [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && echo cuda || echo cpu )"
  if [[ -d data/a2sl && ! -f data/features/a2sl_resnet18.npz ]]; then "$PY" scripts/data/extract_features.py --data a2sl --out data/features/a2sl_resnet18.npz --device cpu; fi
  "$PY" scripts/data/make_splits.py data=avocado
  for g in T10 T20 Tamb; do "$PY" scripts/data/make_splits.py data=avocado splits.mode=logo splits.held_out=$g; done
}

stage_synthetic() { JOBS[synthetic]=$(submit synthetic "$PY scripts/run/synthetic_recovery.py"); }

stage_e1() {
  for m in "${MODELS[@]}"; do JOBS[e1_$m]=$(submit "e1-$m" "$PY scripts/run/train.py model=$m data=avocado"); done
  JOBS[e1_labels]=$(submit "e1-labels-only" "$PY scripts/run/train.py model=latent_sde data=avocado +experiment=labels_only")
}

stage_e2() {
  JOBS[e2_pde]=$(submit e2-pde "$PY scripts/run/train.py model=latent_sde data=avocado model.event_likelihood=pde run_name=latent_sde_avocado_pde")
  JOBS[e2_mc]=$(submit e2-mc "$PY scripts/run/train.py model=latent_sde data=avocado model.event_likelihood=mc run_name=latent_sde_avocado_mc")
  JOBS[e2_filter]=$(submit e2-filter "$PY scripts/run/train.py model=fpt_filter data=avocado run_name=fpt_filter_avocado_exact")
  JOBS[e2_synth_filter]=$(submit e2-synth-filter "$PY scripts/run/train.py model=fpt_filter data=synthetic run_name=fpt_filter_synthetic_exact")
  JOBS[e2_synth_pde]=$(submit e2-synth-pde "$PY scripts/run/train.py model=latent_sde data=synthetic model.event_likelihood=pde run_name=latent_sde_synthetic_pde")
  JOBS[e2_synth_mc]=$(submit e2-synth-mc "$PY scripts/run/train.py model=latent_sde data=synthetic model.event_likelihood=mc run_name=latent_sde_synthetic_mc")
}

stage_e3() {
  for g in T10 T20 Tamb; do
    for m in direct_reg gru_d latent_sde fpt_filter; do JOBS[e3_${m}_$g]=$(submit "e3-$m-$g" "$PY scripts/run/train.py model=$m data=avocado +experiment=logo splits.held_out=$g"); done
    for c in group fixed; do JOBS[e3_clock_${c}_$g]=$(submit "e3-clock-$c-$g" "$PY scripts/run/train.py model=latent_sde data=avocado +experiment=logo splits.held_out=$g model.clock=$c run_name=latent_sde_avocado_logo_${g}_clock_$c"); done
  done
}

stage_e4() {
  for k in 2 3; do for m in gru_d latent_ode latent_sde; do JOBS[e4_${m}_$k]=$(submit "e4-$m-every$k" "$PY scripts/run/train.py model=$m data=avocado +experiment=sampling_every$k"); done; done
  # photo-count curves come from e1's metrics_by_n_obs.json; evaluating daily-trained models on sparser visits:
  local deps=(); for m in gru_d latent_ode latent_sde; do [[ -n "${JOBS[e1_$m]:-}" && "${JOBS[e1_$m]}" != local ]] && deps+=("${JOBS[e1_$m]}"); done
  JOBS[e4_eval]=$(submit e4-eval "for m in gru_d latent_ode latent_sde; do for k in 2 3; do $PY scripts/run/evaluate.py outputs/runs/\${m}_avocado --every \$k; done; done" "${deps[@]}")
}

stage_e6() {
  # the manifests are committed but the raw A2SL data (scans, photos) is not; without it e6 would fail and block summary
  [[ -f data/a2sl/scans_avocado_jun2023.npz ]] || { echo "e6 skipped: data/a2sl not present (docs/data/data_a2sl.md)" >&2; return; }
  for m in gru_d latent_sde fpt_filter; do JOBS[e6_$m]=$(submit "e6-$m" "$PY scripts/run/train.py model=$m data=a2sl horizon=20 train.batch_size=8"); done
  JOBS[e6_gen]=$(submit e6-generative "$PY scripts/run/train.py model=latent_sde data=a2sl horizon=20 train.batch_size=8 model.decode_features=true run_name=latent_sde_a2sl_generative")
}

# Independent multi-seed rerun of every avocado experiment (e1-e4), written to outputs/avocado/round1/seed<s>/<run>
# so outputs/runs is never touched. One job per run, looping over seeds; RIPEN_SEEDS overrides the seed list.
AVOCADO_RUNS=()
for m in "${MODELS[@]}"; do AVOCADO_RUNS+=("model=$m"); done
AVOCADO_RUNS+=("model=latent_sde +experiment=labels_only"
  "model=latent_sde model.event_likelihood=pde run_name=latent_sde_avocado_pde"
  "model=latent_sde model.event_likelihood=mc run_name=latent_sde_avocado_mc"
  "model=fpt_filter run_name=fpt_filter_avocado_exact")
for g in T10 T20 Tamb; do
  for m in direct_reg gru_d latent_sde fpt_filter; do AVOCADO_RUNS+=("model=$m +experiment=logo splits.held_out=$g"); done
  for c in group fixed; do AVOCADO_RUNS+=("model=latent_sde +experiment=logo splits.held_out=$g model.clock=$c run_name=latent_sde_avocado_logo_${g}_clock_$c"); done
done
for k in 2 3; do for m in gru_d latent_ode latent_sde; do AVOCADO_RUNS+=("model=$m +experiment=sampling_every$k"); done; done

stage_seeds() {
  local seeds="${RIPEN_SEEDS:-1 2 3 4 5}" i=0 deps=()
  for args in "${AVOCADO_RUNS[@]}"; do
    local data="data=avocado"; [[ " $args " == *" data="* ]] && data=""
    local cmd="fail=0; for s in $seeds; do $PY scripts/run/train.py $data $args seed=\$s +train.seed=\$s 'out_dir=outputs/avocado/round1/seed'\$s'/\${run_name}' || fail=1; done; exit \$fail"
    JOBS[seeds_$i]=$(submit "seeds-$i" "$cmd"); [[ "${JOBS[seeds_$i]}" != local ]] && deps+=("${JOBS[seeds_$i]}"); i=$((i + 1))
  done
  # afterany: one failed seed should not block the table for the rest
  if [[ $LOCAL -eq 1 ]]; then "$PY" scripts/analysis/summarize_seeds.py; else
    local dep=""; [[ ${#deps[@]} -gt 0 ]] && dep="--dependency=afterany:$(IFS=:; echo "${deps[*]}")"
    JOBS[seeds_summary]=$(sbatch --parsable --job-name=ripen-seeds-summary --output="outputs/logs/%x-%j.out" --time=00:30:00 \
      --cpus-per-task=1 --mem=4G --account="$RIPEN_ACCOUNT" --partition="$RIPEN_PARTITION" --qos="$RIPEN_QOS" $dep \
      --wrap="cd '$RIPEN_ROOT' && source '$RIPEN_ENV/bin/activate' && $PY scripts/analysis/summarize_seeds.py")
  fi
}

# Round 2: the latent SDE with the guided posterior (E0 fix), ResNet-18 features, in-distribution + held-out groups.
R2_GUIDE="model.posterior_guidance=true model.event_mean=log_mean"
R2_RUNS=("model=latent_sde $R2_GUIDE run_name=latent_sde_avocado_guided_r18")
for g in T10 T20 Tamb; do
  R2_RUNS+=("model=latent_sde +experiment=logo splits.held_out=$g $R2_GUIDE run_name=latent_sde_avocado_logo_${g}_guided_r18")
done
# Round 2b: the rest of E2-E4 and E6 for the main model (guided, ResNet-18), same root, same seeds
R2B_RUNS=("model=latent_sde $R2_GUIDE +experiment=labels_only run_name=latent_sde_avocado_labels_only_guided_r18"
  "model=latent_sde $R2_GUIDE model.event_likelihood=mc run_name=latent_sde_avocado_mc_guided_r18")
for g in T10 T20 Tamb; do for c in group fixed; do
  R2B_RUNS+=("model=latent_sde $R2_GUIDE +experiment=logo splits.held_out=$g model.clock=$c run_name=latent_sde_avocado_logo_${g}_clock_${c}_guided_r18")
done; done
for k in 2 3; do R2B_RUNS+=("model=latent_sde $R2_GUIDE +experiment=sampling_every$k run_name=latent_sde_avocado_every${k}_guided_r18"); done
A2SL_ARGS="data=a2sl horizon=20 train.batch_size=8"
R2B_RUNS+=("model=latent_sde $A2SL_ARGS $R2_GUIDE run_name=latent_sde_a2sl_guided_r18"
  "model=gru_d $A2SL_ARGS" "model=fpt_filter $A2SL_ARGS")

# New baselines (classical degradation + landmark Cox; neural CDE + transformer): pilot/sweep on seed 1, then 5 seeds
# on E1, E3 (3 held-out groups), E4 (every 2nd / 3rd day) and A2SL -> outputs/avocado/baselines/
read -r -a NB_MODELS <<< "${RIPEN_NB_MODELS:-wiener gamma_process landmark_cox neural_cde transformer}"
NB_SWEEP=("model=neural_cde run_name=sweep_cde_h32" "model=neural_cde model.hidden=64 model.visit_dim=32 run_name=sweep_cde_h64"
  "model=neural_cde train.lr=3e-4 run_name=sweep_cde_h32_lr3e-4"
  "model=transformer run_name=sweep_tf_d64_l2" "model=transformer model.d_model=32 model.layers=1 run_name=sweep_tf_d32_l1"
  "model=transformer train.lr=3e-4 run_name=sweep_tf_d64_l2_lr3e-4" "model=transformer model.d_model=64 model.layers=3 run_name=sweep_tf_d64_l3")
# the classical models have a handful of parameters: they need a larger step and more epochs (60 at 1e-3 had not converged)
for m in wiener gamma_process landmark_cox; do for lr in 1e-3 1e-2 3e-2; do
  NB_SWEEP+=("model=$m train.lr=$lr train.epochs=200 train.patience=30 run_name=sweep_${m}_lr$lr"); done; done
# chosen by val CRPS on the seed-1 sweep (outputs/avocado/baselines_sweep/): CDE hidden 64; transformer d=32, 1 layer
NB_ARGS_neural_cde="${NB_ARGS_neural_cde:-model.hidden=64 model.visit_dim=32}"
NB_ARGS_transformer="${NB_ARGS_transformer:-model.d_model=32 model.layers=1}"
NB_ARGS_landmark_cox="${NB_ARGS_landmark_cox:-train.lr=3e-2 train.epochs=200 train.patience=30}"
NB_ARGS_gamma_process="${NB_ARGS_gamma_process:-train.lr=3e-2 train.epochs=200 train.patience=30}"
NB_ARGS_wiener="${NB_ARGS_wiener:-}"   # default lr/epochs had the best val CRPS (1.33 vs 1.36-1.37)
nb_runs() {
  NB_RUNS=()
  for m in "${NB_MODELS[@]}"; do
    local extra_var="NB_ARGS_$m" x; x="${!extra_var:-}"
    NB_RUNS+=("model=$m $x")
    for g in T10 T20 Tamb; do NB_RUNS+=("model=$m $x +experiment=logo splits.held_out=$g"); done
    for k in 2 3; do NB_RUNS+=("model=$m $x +experiment=sampling_every$k"); done
    NB_RUNS+=("model=$m $x data=a2sl horizon=20 train.batch_size=8")
  done
}

# seed_jobs TAG ROOT run-args... : one job per run looping over RIPEN_SEEDS, then a summary job (afterany)
seed_jobs() {
  local tag="$1" root="$2"; shift 2
  local seeds="${RIPEN_SEEDS:-1 2 3 4 5}" i=0 deps=()
  for args in "$@"; do
    local data="data=avocado"; [[ " $args " == *" data="* ]] && data=""
    local cmd="fail=0; for s in $seeds; do $PY scripts/run/train.py $data $args seed=\$s +train.seed=\$s 'out_dir=$root/seed'\$s'/\${run_name}' || fail=1; done; exit \$fail"
    JOBS[${tag}_$i]=$(submit "$tag-$i" "$cmd"); [[ "${JOBS[${tag}_$i]}" != local ]] && deps+=("${JOBS[${tag}_$i]}"); i=$((i + 1))
  done
  local post="$PY scripts/analysis/recalibrate.py $root && $PY scripts/analysis/summarize_seeds.py --root $root ${RIPEN_SUMMARY_ARGS:-} && $PY scripts/analysis/summarize_seeds.py --root $root --recal ${RIPEN_SUMMARY_ARGS:-}"
  if [[ $LOCAL -eq 1 ]]; then bash -c "$post"; else
    local dep=""; [[ ${#deps[@]} -gt 0 ]] && dep="--dependency=afterany:$(IFS=:; echo "${deps[*]}")"
    JOBS[${tag}_summary]=$(sbatch --parsable --job-name="ripen-$tag-summary" --output="outputs/logs/%x-%j.out" --time=00:30:00 \
      --cpus-per-task=1 --mem=4G --account="$RIPEN_ACCOUNT" --partition="$RIPEN_PARTITION" --qos="$RIPEN_QOS" $dep \
      --wrap="cd '$RIPEN_ROOT' && source '$RIPEN_ENV/bin/activate' && $post")
  fi
}

stage_r2() { RIPEN_SUMMARY_ARGS="--suffix _guided_r18" seed_jobs r2 outputs/avocado/round2 "${R2_RUNS[@]}"; }
stage_r2b() { RIPEN_SUMMARY_ARGS="--suffix _guided_r18" seed_jobs r2b outputs/avocado/round2 "${R2B_RUNS[@]}"; }
# the new baselines are small: run them on CPU (RIPEN_GPU=none, JAX_PLATFORMS=cpu) so they don't queue behind GPU jobs
stage_nb_sweep() {
  local runs=("${NB_SWEEP[@]}"); [[ -n "${RIPEN_SWEEP_FILTER:-}" ]] && mapfile -t runs < <(printf '%s\n' "${NB_SWEEP[@]}" | grep -E "$RIPEN_SWEEP_FILTER")
  RIPEN_MEM=32G RIPEN_GPU=none JAX_PLATFORMS=cpu RIPEN_SEEDS=1 seed_jobs nbsweep outputs/avocado/baselines_sweep "${runs[@]}"
}
stage_nb() { nb_runs; RIPEN_MEM=32G RIPEN_GPU=none JAX_PLATFORMS=cpu seed_jobs nb outputs/avocado/baselines "${NB_RUNS[@]}"; }

# Mango study (docs: docs/experiments/2026-11-06_mango-photo-vs-microwave/report.md): CPU jobs, one per model x input setting, each looping 3 seeds x 6 folds
# (one process per seed: XLA's CPU code cache grows with every fold's freshly-jitted train step)
MANGO_MODELS=(${RIPEN_MANGO_MODELS:-gru_gap gru_d ode_rnn neural_cde transformer latent_sde latent_ode fpt_filter})
MANGO_SETTINGS=(${RIPEN_MANGO_SETTINGS:-rgb mw both samd})
stage_mango_data() {
  JOBS[mango_data]=$(RIPEN_GPU=none submit mango-data \
    "$PY scripts/data/ingest_mango.py && $PY scripts/data/extract_features.py --data mango --out data/features/mango_resnet18.npz")
}
stage_mango() {
  local deps=() m s
  JOBS[mango_refs]=$(RIPEN_GPU=none submit mango-refs "$PY scripts/run/mango_cv.py --model refs --seeds 1 2 3"); deps+=("${JOBS[mango_refs]}")
  for m in "${MANGO_MODELS[@]}"; do for s in "${MANGO_SETTINGS[@]}"; do
    JOBS[mango_${m}_$s]=$(RIPEN_GPU=none submit "mango-$m-$s" \
      "for seed in 1 2 3; do JAX_PLATFORMS=cpu $PY scripts/run/mango_cv.py --model $m --setting $s --seeds \$seed || exit 1; done")
    deps+=("${JOBS[mango_${m}_$s]}")
  done; done
  JOBS[mango_summary]=$(sbatch --parsable --job-name=ripen-mango-summary --output="outputs/logs/%x-%j.out" --time=00:30:00 \
    --cpus-per-task=1 --mem=4G --account="$RIPEN_ACCOUNT" --partition="$RIPEN_PARTITION" --qos="$RIPEN_QOS" \
    --dependency=afterany:$(IFS=:; echo "${deps[*]}") \
    --wrap="cd '$RIPEN_ROOT' && source '$RIPEN_ENV/bin/activate' && $PY scripts/analysis/mango_summary.py")
}

# A2SL: which microwave view predicts best (docs/experiments/2026-11-10_avocado-microwave-scanner-views/report.md). One job per model x input x view, each looping
# 3 seeds x 4 folds; inputs mw / photo_mw at every view (0..300, six pooled, six side by side), plus photo only and the
# no-sensor control. RIPEN_VIEW_LABELS=input reruns the presented setup (labels as inputs); default hidden. Jobs alternate
# GPU / 2-CPU so more run at once under the per-user caps (4 GPUs, 16 CPUs). RIPEN_SCAN_STD=1: z-scored microwave
# numbers (round 3), into labels_<mode>_std (and outputs/a2sl_aml_std for a2sl-aml).
VIEW_MODELS=(${RIPEN_VIEW_MODELS:-latent_sde fpt_filter})
stage_a2sl_views() {
  local deps=() m i v j n=0 lab="${RIPEN_VIEW_LABELS:-hidden}" args=() std="" sfx=""
  if [[ "${RIPEN_SCAN_STD:-0}" == 1 ]]; then std="data.scan_standardize=true "; sfx="_std"; fi
  for m in "${VIEW_MODELS[@]}"; do
    for i in mw photo_mw; do for v in all concat 0+120+240 60+180+300 0 60 120 180 240 300; do args+=("--model $m --input $i --view $v"); done; done
    # photo-only / time-only runs do not read scans: in the _std root reuse the raw-root runs when they exist
    for i in photo time; do [[ -z "$sfx" || ! -d outputs/a2sl_views/labels_$lab/seed1/${m}_$i ]] && args+=("--model $m --input $i"); done
  done
  local root="outputs/a2sl_views/labels_$lab$sfx"
  for a in "${args[@]}"; do
    local name="views-${lab}${sfx}-$(echo "$a" | awk '{print $2"-"$4"-"$6}')"
    if (( n % 2 )); then
      j=$(RIPEN_GPU=none RIPEN_CPUS=2 submit "$name" "JAX_PLATFORMS=cpu $PY scripts/run/a2sl_views.py $std$a --labels $lab --out $root --seeds 1 2 3")
    else
      j=$(submit "$name" "$PY scripts/run/a2sl_views.py $std$a --labels $lab --out $root --seeds 1 2 3")
    fi
    deps+=("$j"); n=$((n + 1))
  done
  if [[ -n "$sfx" ]]; then  # photo-only and time-only runs do not read scans: reuse them
    for m in "${VIEW_MODELS[@]}"; do for d in outputs/a2sl_views/labels_$lab/seed*/${m}_photo outputs/a2sl_views/labels_$lab/seed*/${m}_time; do [[ -d "$d" ]] || continue
      mkdir -p "$root/$(basename "$(dirname "$d")")"; ln -sfn "$(realpath "$d")" "$root/$(basename "$(dirname "$d")")/$(basename "$d")"
    done; done
  fi
  JOBS[views_summary]=$(sbatch --parsable --job-name=ripen-views-summary --output="outputs/logs/%x-%j.out" --time=00:30:00 \
    --cpus-per-task=1 --mem=8G --account="$RIPEN_ACCOUNT" --partition="$RIPEN_PARTITION" --qos="$RIPEN_QOS" \
    --dependency=afterany:$(IFS=:; echo "${deps[*]}") \
    --wrap="cd '$RIPEN_ROOT' && source '$RIPEN_ENV/bin/activate' && export JAX_PLATFORMS=cpu && $PY scripts/analysis/a2sl_views_summary.py --root $root && $PY scripts/analysis/a2sl_views_summary.py --root $root --recal")
  JOBS[views_n]=${#deps[@]}
}

# AML on A2SL (docs/experiments/2026-11-10_avocado-microwave-scanner-views/report.md): photo + microwave teacher (RIPEN_AML_VIEW, default all = six pooled, the best photo + microwave input in the hidden-label view study) -> photo-only
# students; one job per model x method, 3 seeds x 4 folds; alternating GPU / 2-CPU like a2sl-views
stage_a2sl_aml() {
  local deps=() m me j n=0 v="${RIPEN_AML_VIEW:-all}" args=() std="" out="outputs/a2sl_aml"
  if [[ "${RIPEN_SCAN_STD:-0}" == 1 ]]; then std="data.scan_standardize=true "; out="outputs/a2sl_aml_std"; fi
  for m in "${VIEW_MODELS[@]}"; do
    for me in ${RIPEN_AML_METHODS:-photo teacher kd feat kd_feat samd}; do args+=("--model $m --method $me"); done
    [[ -z "${RIPEN_AML_METHODS:-}" ]] && args+=("--model $m --method kd --alpha 0.5 --temp 1 --gamma 0.5 --tag _a05_t1")
    while IFS= read -r x; do [[ -n "$x" ]] && args+=("--model $m $x"); done <<< "${RIPEN_AML_EXTRA:-}"  # one variant per line
  done
  for a in "${args[@]}"; do
    local name="aml-$(echo "$a" | awk '{print $2"-"$4}')$(echo "$a" | grep -o -- '--tag [^ ]*' | cut -d' ' -f2 || true)"
    if (( n % 2 )); then
      j=$(RIPEN_GPU=none RIPEN_CPUS=2 submit "$name" "JAX_PLATFORMS=cpu $PY scripts/run/a2sl_aml.py $std$a --view $v --out $out --seeds 1 2 3")
    else
      j=$(submit "$name" "$PY scripts/run/a2sl_aml.py $std$a --view $v --out $out --seeds 1 2 3")
    fi
    deps+=("$j"); n=$((n + 1))
  done
  JOBS[aml_summary]=$(sbatch --parsable --job-name=ripen-aml-summary --output="outputs/logs/%x-%j.out" --time=00:30:00 \
    --cpus-per-task=1 --mem=8G --account="$RIPEN_ACCOUNT" --partition="$RIPEN_PARTITION" --qos="$RIPEN_QOS" \
    --dependency=afterany:$(IFS=:; echo "${deps[*]}") \
    --wrap="cd '$RIPEN_ROOT' && source '$RIPEN_ENV/bin/activate' && export JAX_PLATFORMS=cpu && $PY scripts/analysis/a2sl_aml_summary.py --root $out && $PY scripts/analysis/a2sl_aml_summary.py --root $out --recal")
  JOBS[aml_n]=${#deps[@]}
}

# mango views / time series / AML (docs/plans/views_aml_plan.md): RIPEN_MANGO_RUNS holds one mango_cv.py argument
# list per line, each its own 2-CPU job, into outputs/mango_views; RIPEN_MANGO_GPU=1 puts them on a GPU instead
stage_mango_views() {
  local deps=() j x
  while IFS= read -r x; do
    [[ -z "$x" ]] && continue
    local name="mv-$(echo "$x" | tr -s ' ' '-' | tr -d '+' | cut -c1-60)"
    if [[ "${RIPEN_MANGO_GPU:-0}" == 1 ]]; then
      j=$(submit "$name" "$PY scripts/run/mango_cv.py $x --out outputs/mango_views --seeds 1 2 3")
    else
      j=$(RIPEN_GPU=none RIPEN_CPUS=2 submit "$name" "JAX_PLATFORMS=cpu $PY scripts/run/mango_cv.py $x --out outputs/mango_views --seeds 1 2 3")
    fi
    deps+=("$j")
  done <<< "${RIPEN_MANGO_RUNS:-}"
  JOBS[mango_views_n]=${#deps[@]}
}

stage_recal() {
  JOBS[recal]=$(submit recal "$PY scripts/analysis/recalibrate.py outputs/avocado/round1 && $PY scripts/analysis/summarize_seeds.py --recal")
}

stage_summary() {
  local deps=(); for j in "${JOBS[@]}"; do [[ "$j" != local ]] && deps+=("$j"); done
  JOBS[summary]=$(submit summary "$PY scripts/analysis/summarize.py" "${deps[@]}")
}

case "$STAGE" in
  setup) if [[ $LOCAL -eq 1 ]]; then stage_setup; else self_job setup 02:00:00; fi ;;
  data) if [[ $LOCAL -eq 1 ]]; then stage_data; else self_job data 04:00:00; fi ;;
  pipeline)
    [[ $LOCAL -eq 1 ]] && die "pipeline is the sbatch chain; use setup/data/all --local instead"
    self_job setup 02:00:00; self_job data 04:00:00 "${JOBS[setup]}"; AFTER+=("${JOBS[data]}")
    stage_synthetic; stage_e1; stage_e2; stage_e3; stage_e4; stage_e6; stage_summary ;;
  synthetic) stage_synthetic ;;
  e1) stage_e1 ;;
  e2) stage_e2 ;;
  e3) stage_e3 ;;
  e4) stage_e4 ;;
  e6) stage_e6 ;;
  summary) stage_summary ;;
  seeds) stage_seeds ;;
  seeds-summary) "$PY" scripts/analysis/summarize_seeds.py ;;
  r2) stage_r2 ;;
  r2b) stage_r2b ;;
  nb-sweep) stage_nb_sweep ;;
  nb) stage_nb ;;
  recal) stage_recal ;;
  mango-data) stage_mango_data ;;
  mango) stage_mango ;;
  mango-views) stage_mango_views ;;
  a2sl-views) stage_a2sl_views ;;
  a2sl-aml) stage_a2sl_aml ;;
  all) stage_synthetic; stage_e1; stage_e2; stage_e3; stage_e4; stage_e6; stage_summary ;;
  *) die "unknown stage $STAGE" ;;
esac
[[ $LOCAL -eq 1 ]] || { echo "submitted:"; for k in "${!JOBS[@]}"; do echo "  $k -> ${JOBS[$k]}"; done | sort; }
