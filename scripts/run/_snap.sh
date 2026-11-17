# shellcheck shell=bash
# Source from a submit script: jobs run from a frozen git worktree of HEAD, so editing the repo while they run cannot
# change their code (the drivers start a fresh train.py per fold, which re-imports ripen). data/, outputs/ and
# .ripen.env are symlinked from the repo; PYTHONPATH puts the snapshot's src first, ahead of the editable install.
# Commit first: uncommitted changes are not in the snapshot.
REPO="$(git rev-parse --show-toplevel)"
SHA="$(git -C "$REPO" rev-parse --short HEAD)"
SNAP="$(dirname "$REPO")/produce-sde-runs/$SHA"
if [[ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]]; then echo "warning: uncommitted changes are not in snapshot $SHA" >&2; fi
if [[ ! -d "$SNAP" ]]; then
  git -C "$REPO" worktree add --detach -q "$SNAP" HEAD
  rm -rf "$SNAP/data"
  ln -s "$REPO/data" "$SNAP/data"; ln -s "$REPO/outputs" "$SNAP/outputs"; ln -s "$REPO/.ripen.env" "$SNAP/.ripen.env"
fi
SNAP_ENV="cd '$SNAP' && source ./.ripen.env && source \$RIPEN_ENV/bin/activate && export PYTHONPATH='$SNAP/src' JAX_PLATFORMS=cpu"
# sub NAME "command" [sbatch args...]  -> job id; runs in the snapshot, 2 CPUs / 16 GB / 12 h unless overridden
sub() { local name="$1" cmd="$2"; shift 2
  sbatch --parsable -p class -A class -q default -c 2 --mem=16G -t 12:00:00 -J "ripen-$name" \
    -o "$REPO/outputs/logs/%x-%j.out" "$@" --wrap "$SNAP_ENV && $cmd"; }
