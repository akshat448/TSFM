# PATHS_AND_MIGRATION.md

One-time migration from the bridge phase's college DGX server
(`/workspace/TSFM/`, single A100) to the new server
(`/mnt/hdd1/TSFM/`, 8x RTX PRO 6000 Blackwell 97GB). Run this once, then it's
reference-only.

## 0. What's actually moving

From the old server:
- `/workspace/TSFM/` (the whole repo: `config/`, `scripts/`, `notebooks/`,
  `literature/`, `documents/`, `datasets/`) — small, ~2.4GB total including
  data (per the final STATUS.md entry: 1.5GB raw + 900MB processed).
- The WESAD symlink target does **not** need to move — it pointed at
  `/workspace/TimeSeriesBenchMark/datasets/WESAD`, which lives on the *old*
  server and won't exist on the new one. WESAD's actual bytes need a real
  copy this time (see step 2), the new server doesn't have that other lab's
  directory to symlink against.
- Nothing else — MIMIC-IV clinical tables and the five inventoried lab
  directories (`Emotiv`, `Brain`, `SensorLLM`, `GlycoLM`, `TSZechen`) belong
  to other people's projects on the old server and are out of scope for this
  migration entirely.

## 1. New directory layout

```
/mnt/hdd1/TSFM/
├── codebase/     <- git repo: config/, scripts/, notebooks/, literature/, documents/, *.md
├── dataset/      <- raw/ and processed/ (was datasets/ nested inside the repo before — now a sibling)
├── models/       <- not created yet, checkpoint cache, first task creates it
└── venv/         <- new uv-managed venv, sibling of the above (was /workspace/Venv/TSFM before)
```

Run this once to scaffold it:

```bash
mkdir -p /mnt/hdd1/TSFM/codebase
mkdir -p /mnt/hdd1/TSFM/dataset/raw/{energy,traffic,weather,ecg,multi_domain,wearables}
mkdir -p /mnt/hdd1/TSFM/dataset/processed/{energy,traffic,weather,ecg,multi_domain,wearables}
mkdir -p /mnt/hdd1/TSFM/dataset/manifests
mkdir -p /mnt/hdd1/TSFM/models/checkpoints
mkdir -p /mnt/hdd1/TSFM/models/cache/huggingface
```
(`setup_new_server.sh` in this bundle does this plus the codebase subtree in
one shot.)

## 2. Copy the repo and data over

Run **from the new server**, pulling from the old one (adjust
`OLD_HOST`/`OLD_USER` — whatever you SSH into the DGX box as):

```bash
export OLD=OLD_USER@OLD_HOST:/workspace/TSFM

# repo/code (everything except datasets/, which moves to dataset/ separately)
rsync -avz --progress \
  --exclude 'datasets/' \
  --exclude '.cache/' \
  --exclude '__pycache__/' \
  "$OLD"/ /mnt/hdd1/TSFM/codebase/

# data: bridge-phase datasets/raw and datasets/processed -> new dataset/
rsync -avz --progress "$OLD"/datasets/raw/      /mnt/hdd1/TSFM/dataset/raw/
rsync -avz --progress "$OLD"/datasets/processed/ /mnt/hdd1/TSFM/dataset/processed/
rsync -avz --progress "$OLD"/datasets/manifests/ /mnt/hdd1/TSFM/dataset/manifests/
```

**WESAD needs a real copy this time**, not a re-created symlink (the old
symlink's target, `/workspace/TimeSeriesBenchMark/`, doesn't exist on this
box):

```bash
rsync -avz --progress OLD_USER@OLD_HOST:/workspace/TimeSeriesBenchMark/datasets/WESAD/ \
  /mnt/hdd1/TSFM/dataset/raw/wearables/wesad/
```

If the old server is unreachable via direct `rsync` (no SSH between the two
boxes), fall back to `scp`/`tar` through your local machine, or re-download
WESAD fresh from its own PhysioNet-adjacent source (UCI ML repo hosts it too)
rather than treating the old server as the only copy — check with the user
before re-downloading anything that took real time to get before.

## 3. New venv

```bash
cd /mnt/hdd1/TSFM
uv venv venv --python 3.11
source venv/bin/activate
python --version   # confirm 3.11.x
uv pip install datasets huggingface_hub transformers numpy pandas scikit-learn \
  wfdb pyarrow tqdm requests openpyxl matplotlib seaborn ipykernel gluonts gdown pyyaml
# torch: check CUDA 13.2 compatibility on this box before installing the same
# cu126 build the old server used — a newer wheel may be needed. This can
# wait until a GPU slot is actually being used; CPU-only torch is fine for
# now if you want it installed early for adapter smoke tests.
uv pip install torch torchvision torchaudio
```

## 4. New `config/paths.yaml`

Replace the bridge phase's `paths.yaml` with the version in this bundle
(`config_paths.yaml` — copy it to `codebase/config/paths.yaml`). Key
differences from the old file:
- `data_root` now points outside the repo (`/mnt/hdd1/TSFM/dataset`), not a
  `datasets/` subfolder inside it.
- New `checkpoint_root` key for `models/`, didn't exist in the bridge phase
  (no model downloads happened then).
- `existing_assets_root` removed — it pointed at
  `/workspace/TimeSeriesBenchMark`, an old-server-only path. If this new
  server has its own existing-assets directory worth pointing at, add it back
  once `NEXT_PHASE_TASKS.md` task A0 (inventory this box) finds one.
- `venv_path` updated to `/mnt/hdd1/TSFM/venv`.

## 5. Validate the migration

```bash
source /mnt/hdd1/TSFM/venv/bin/activate
cd /mnt/hdd1/TSFM/codebase
python -c "import yaml; c = yaml.safe_load(open('config/paths.yaml')); print(c)"
python -c "
import pandas as pd, yaml
c = yaml.safe_load(open('config/paths.yaml'))
df = pd.read_parquet(f\"{c['processed_dir']}/energy/ETTh1.parquet\")
print(df.shape, df.columns.tolist())
"
find /mnt/hdd1/TSFM/dataset/raw -maxdepth 2 -type d
du -sh /mnt/hdd1/TSFM/dataset/raw /mnt/hdd1/TSFM/dataset/processed
```

Expect: `ETTh1.parquet` loads with the same schema documented in `DATASETS.md`
(`item_id`, `start`, `freq`, `target`, `group_id`, split-index columns), and
`dataset/raw` + `dataset/processed` sum to roughly the same ~2.4GB the bridge
phase reported. If either check fails, stop and diagnose before running any
new downloads on top of a possibly-broken migration.

## 6. grep for anything that still hardcodes the old path

```bash
grep -rn "/workspace/TSFM" /mnt/hdd1/TSFM/codebase --include="*.py" --include="*.yaml" --include="*.md"
grep -rn "/workspace/Venv" /mnt/hdd1/TSFM/codebase --include="*.py"
```

The bridge phase's portability audit (STATUS.md, 2026-07-08) already found
and fixed one such violation (`standardize_monash.py`). This grep should come
back empty; if it doesn't, fix it the same way — remove the hardcoded path,
rely on `config/paths.yaml` or the activated venv instead.