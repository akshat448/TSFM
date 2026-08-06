# NEXT_PHASE_TASKS.md

Ordered checklist for this phase. Bucket A needs no GPU and can start
immediately. Bucket B needs an active, confirmed GPU slot — don't start it
early even if a GPU looks briefly idle in `nvidia-smi` (per `AGENTS.md` rule
2, other users' jobs can resume). Each task below has a matching ready-to-paste
prompt in `CODEX_PROMPTS.md`.

---

## Bucket A — do now, no GPU required

### A0. Inventory this new server for existing assets
Before downloading anything, check whether `/mnt/hdd1/` or elsewhere on this
box already has TSFM-relevant data, the way the old DGX server turned out to
have MIMIC-IV clinical and WESAD sitting around. Don't assume a clean slate.
→ `CODEX_PROMPTS.md` Prompt 1.

### A1. Run the migration
Directory scaffolding, rsync from the old server, new venv, new
`config/paths.yaml`, validation. See `PATHS_AND_MIGRATION.md` in full.
→ `CODEX_PROMPTS.md` Prompt 2.

### A2. Deploy and dry-run the harness (CPU-only checks)
Copy `Phase_1_Harness` into `codebase/`, update its config to the new
`paths.yaml`, confirm `run_phase1.py --dataset ETTh1 --model dlinear`
actually runs the plumbing end-to-end (real reproduction-gate numbers need a
GPU-adjacent slot for DLinear training to finish in reasonable time, but the
import/data-loading/metrics/output-writing path should be verified now).
→ `CODEX_PROMPTS.md` Prompt 3.

### A3. Expand datasets to the full Recommended tier + start Full-tier prep
This server's storage (check `df -h`, but 8x97GB GPU boxes are usually paired
with real disk) supports going past the bridge phase's deliberately narrow
6-8GB target. In scope now:
- **Full GIFT-Eval** (all 23 datasets/7 domains, ~5GB) instead of the
  3-config subset.
- **Expanded Monash** beyond the 20 chosen, if useful — or at minimum
  confirm the 20 already chosen are still the right set now that scale isn't
  a constraint.
- **PPG-DaLiA** (UCI ML Repository, direct headless zip) — listed as
  "team extension, not selected this phase" in the bridge phase, now in
  scope.
- **EEG (TUAB/TUEV)** — was Full-tier/Optional per `BENCHMARK_SPEC.md`,
  worth a scoping conversation now that compute isn't the constraint, but
  confirm with the user before starting (`BENCHMARK_SPEC.md`'s domain
  verdict table still marks it optional, "only if ECG lands ahead of
  schedule" — it did, ECG/PTB-XL is done).
- **Start MIMIC-IV-ECG credentialing** — this is a paperwork/registration
  task (PhysioNet credentialed-data application), not a download task yet.
  Needs the user's institutional affiliation info, don't do this silently.
→ `CODEX_PROMPTS.md` Prompt 4.

### A4. Pre-stage every model checkpoint the Phase 1 model set needs
Downloading weights needs disk + network, not GPU compute. Get ahead of the
GPU slot by pre-downloading everything in `PHASE1_PLAN.md`'s Wave 1-4 list
now, cached under `models/checkpoints` per the new `paths.yaml`. This is the
single highest-leverage no-GPU task, it means Bucket B starts running
experiments immediately instead of waiting on downloads.
→ `CODEX_PROMPTS.md` Prompt 5.

### A5. Verify the three Wave 2 adapter stubs against real (downloaded) weights, CPU-only where feasible
`chronos_bolt.py`, `timesfm.py`, `ttm.py` in the harness are untested
scaffolding (per the harness README). TTM-r2 is small enough to plausibly
smoke-test on CPU. Chronos-Bolt-base and TimesFM-2.5 may be CPU-feasible for
a single-series smoke test even if too slow for a real sweep. Get as far as
possible without a GPU; flag whatever genuinely needs one for Bucket B.
→ `CODEX_PROMPTS.md` Prompt 6.

### A6. Build the cross-dataset contamination/dedup scanner
`STATUS.md`'s final entry explicitly recommends this as the first Phase 1
priority: generalize the ad hoc solar/electricity correlation checks from the
bridge phase into a reusable tool that runs pairwise across every dataset in
the collection, especially once A3 adds more datasets. No GPU needed, this is
pure CPU/pandas work, and it directly feeds `contamination_matrix.yaml`.
→ `CODEX_PROMPTS.md` Prompt 7.

### A7. Read Meyer et al. 2025 (arXiv:2510.13654) and update the leakage matrix
Flagged as a must-read in `STATUS.md` before Phase 2's leakage matrix design
starts. Worth doing now, before the first real model runs, since it may
change which contamination labels get attached to Wave 2+ results from day
one.
→ `CODEX_PROMPTS.md` Prompt 8.

---

## Bucket B — needs a confirmed, active GPU slot

Do not start these until the conversation has an explicit go-ahead that a
slot is live. Sequenced per `PHASE1_PLAN.md` section 6, adjusted for 8 GPUs
instead of 1 (parallelism opportunities noted, not yet decided — flag and
confirm before parallelizing across GPUs rather than assuming).

### B1. M1 — reproduction gate (blocking, do not skip)
Seasonal Naive + DLinear on ETTh1, all four horizons, compare against
published DLinear/LTSF-Linear numbers. If it doesn't match within tolerance,
stop and debug before anything else runs. This is the same gate
`PHASE1_PLAN.md` and `run_phase1.py`'s own docstring already specify.

### B2. M2 — full non-negotiable-baseline table
Seasonal Naive + DLinear across all Minimal-tier forecasting datasets.

### B3. M3 — Wave 2 zero-shot
Chronos-Bolt, TimesFM-2.5, TTM-r2 zero-shot on the same datasets, with
contamination labels attached from the start. This is where A4/A5/A7 pay off
— checkpoints already staged, adapters already smoke-tested, contamination
matrix already current.

### B4. M4 — Recommended-tier extension
Solar, PEMS03/04/08, GIFT-Eval, Monash, using whatever A3 landed.

### B5. M5 — PTB-XL classification + ECG-FM baseline
Needs the classification eval path (already scoped in `PHASE1_PLAN.md`
section 3) and ECG-FM weights (stage in A4). Remember the known
contamination caveat on ECG-FM vs PTB-XL from `STATUS.md` — report it, don't
suppress it, it's headline finding #2 for the whole project.

### B6. Wave 3/4, M6
PatchTST/iTransformer/S-Mamba (trained, not zero-shot), MOMENT/Chronos-2/
Moirai-MoE/GPT4TS/AutoARIMA, then the significance-testing layer. With 8
GPUs available, this is where it's worth an explicit conversation about
running several of these in parallel rather than strictly sequencing waves —
raise it, don't just do it.

---

## Explicitly out of scope until asked

- Actually pulling MIMIC-IV-ECG data (credentialing application only, in A3).
- Few-shot adaptation and fine-tuning regimes (layered on top once zero-shot,
  M3, is solid — `PHASE1_PLAN.md` section 7).
- Anything from CGM/Finance/Server-Telemetry — still `Excluded` per
  `BENCHMARK_SPEC.md`'s domain table, Phase 3 territory.