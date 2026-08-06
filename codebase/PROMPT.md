# CODEX_PROMPTS.md

Copy-paste prompts for separate Codex sessions/tasks. Each assumes Codex has
already read `AGENTS.md` in the repo root (it does this automatically) — the
prompts below only add the task-specific instruction, they don't re-explain
project context that's already in `AGENTS.md`.

Run these roughly in the order they're numbered for a first pass; A0-A2 are
true prerequisites for everything after them.

---

### Prompt 1 — A0: inventory the new server

```
Before any download work, inventory /mnt/hdd1/ and any other obviously
shared data locations on this server (check `df -h`, `ls /mnt/`, ask the
system for other top-level data mounts) for anything that overlaps this
project's dataset scope: ETT, Electricity, Traffic, Weather, PEMS03/04/08,
Solar-Energy, PTB-XL, GIFT-Eval, Monash, or anything from the wearables/EEG/
CGM domains (WESAD, PPG-DaLiA, TUAB/TUEV, MIMIC-IV in any form).

Do this the same way EXISTING_ASSETS.md's bridge-phase inventory was done:
actually inspect file contents/formats, don't guess from directory names.
Write findings into a new EXISTING_ASSETS.md section for this server
(create the file if the migrated copy didn't bring one over, or append a
clearly-dated new section if it did — don't overwrite the old server's
findings, they're still historically relevant even though that data isn't
here). If genuinely nothing overlaps, say so explicitly and move on to the
migration task.
```

---

### Prompt 2 — A1: run the migration

```
Run the migration described in PATHS_AND_MIGRATION.md end to end:
1. Scaffold /mnt/hdd1/TSFM/{codebase,dataset,models,venv} per
   setup_new_server.sh.
2. rsync the old server's repo and datasets/ over (ask me for the old
   server's SSH host/user if it's not already in your context — don't
   guess or invent one).
3. Create the new venv and install the package list from AGENTS.md's
   Environment section.
4. Replace config/paths.yaml with the new version (config_paths.yaml in
   this bundle).
5. Run the validation block in PATHS_AND_MIGRATION.md step 5 and report
   the actual output (parquet shape, file counts, disk usage) — don't just
   report "done," show me it actually loaded real data.
6. Run the grep-for-hardcoded-paths check in step 6 and fix anything it
   finds.

Append a STATUS.md entry when done, same format as the bridge phase's
entries: what got done, what's blocked, what's next.
```

---

### Prompt 3 — A2: deploy and dry-run the harness

```
Copy the Phase_1_Harness bundle's contents into codebase/ per its own
README.md's "where these files go" section. Point phase1_eval.yaml and any
other config at the new config/paths.yaml (don't hardcode the old
/workspace/TSFM path anywhere in it).

Run: python run_phase1.py --dataset ETTh1 --model dlinear

This IS allowed without a GPU slot for now — DLinear is tiny and CPU
training is fine for a plumbing check, we just won't treat the resulting
numbers as the real M1 reproduction-gate result until it's re-run properly
once a GPU slot is confirmed (training-time and any numeric differences from
CPU vs GPU float behavior matter for that gate). Report: did it run without
errors, what were the mean MSE/MASE numbers, and do the raw output rows in
results/phase1/ETTh1.parquet look shaped the way PHASE1_PLAN.md section 3
describes (one row per item_id/horizon/window)?

Do not attempt to run any Wave 2+ model in this task, that needs actual
checkpoints (Prompt 5) and adapter verification (Prompt 6) first.
```

---

### Prompt 4 — A3: expand datasets to full Recommended tier

```
Expand DATASETS.md with new entries (using its existing card template) for:
1. Full GIFT-Eval (all 23 configs, not the 3-config subset the bridge phase
   used) — headless via `huggingface-cli download Salesforce/GiftEval
   --repo-type dataset`. Note in the card how this relates to/supersedes the
   existing gift_eval.md subset card rather than creating a confusing
   duplicate.
2. PPG-DaLiA — UCI ML Repository, confirm a direct headless zip URL exists
   before starting (per CLAUDE.md/AGENTS.md rule 3, headless-only, flag it
   in STATUS.md rather than improvising if it turns out to need a browser).
3. Re-evaluate the 20 chosen Monash sub-datasets now that the <1GB budget
   constraint is gone — either justify keeping the same 20 or propose an
   expanded list, and ask me before actually expanding it (this changes
   scope, per AGENTS.md rule 8, don't decide it silently).

For MIMIC-IV-ECG: do NOT attempt any download. Instead, research and
summarize the actual PhysioNet credentialed-access process for MIMIC-IV-ECG
specifically (required training, typical turnaround, what institutional
info is needed) and write it up as a new section in CLAUDE.md's/AGENTS.md's
known-open-items, with a clear "needs the user's institutional affiliation
and CITI training status" flag rather than attempting to fill that in
yourself.

For EEG (TUAB/TUEV): stop and ask me before starting anything here at all —
BENCHMARK_SPEC.md marks this domain Optional, not Recommended, and I want to
explicitly decide whether to expand scope into it now that compute allows
it, rather than have it happen as a side effect of "we have room now."

Standardize and dataset-card everything you do download, same discipline as
the bridge phase (Steps 3-5 in the old STEPS.md). Update the phase-scope
table at the top of DATASETS.md. Append a STATUS.md entry.
```

---

### Prompt 5 — A4: pre-stage all model checkpoints

```
Pre-download every model checkpoint referenced in PHASE1_PLAN.md's Wave 1-4
model list and BENCHMARK_SPEC.md's ~15-model catalog, into
models/checkpoints per the new checkpoint_root in config/paths.yaml. This is
disk+network work only, no GPU compute involved — do not load any checkpoint
onto a GPU or run inference, just fetch and cache the weights.

For each model: note in a new models/checkpoints/MANIFEST.md (create it)
the exact HF repo ID or source used, the download command, the resulting
size on disk, and the pip package version needed to load it (per
CLAUDE.md's/harness README's package list: chronos-forecasting, timesfm,
granite-tsfm, plus whatever Wave 3/4 need — check each model's own repo for
the current recommended loading library before guessing).

If any model's weights require an EULA click-through or gated HF repo
access (this is common for some LLM-based/domain-FM checkpoints), don't try
to work around it — flag it in STATUS.md with exactly what gate was hit and
move on to the next model rather than spending the session on one blocker,
same discipline as the bridge phase's PEMS03/04/08 handling.

Report total disk used by models/checkpoints/ at the end, and confirm
available disk headroom (df -h) is still healthy given dataset/ is also
growing from the A3 task.
```

---

### Prompt 6 — A5: verify Wave 2 adapter stubs

```
The harness's scripts/harness/models/chronos_bolt.py, timesfm.py, and
ttm.py are untested scaffolding (per the harness README's own admission —
read it first). Now that real checkpoints exist (from Prompt 5's task, run
that first if it hasn't run yet), verify each adapter:

1. Run each file's own __main__ smoke test directly
   (python scripts/harness/models/ttm.py, etc.) against real weights instead
   of synthetic data.
2. For any model small enough to plausibly run on CPU in reasonable time
   (TTM-r2 is the best candidate), actually do a single-series forward pass
   and sanity-check the output shape/values.
3. For models that need a GPU to be practically testable (likely
   Chronos-Bolt and TimesFM-2.5 at real speed), get as far as import +
   checkpoint-loading verification on CPU, then explicitly flag in
   STATUS.md exactly what remains to verify once a GPU slot is active —
   don't claim full verification if you couldn't actually run inference.

Fix any API mismatches you find (the harness README warns these libraries'
APIs move frequently). Do not add any of these to phase1_eval.yaml's
active `models:` block until its smoke test has genuinely passed — leave
newly-verified ones as an uncommented block ready to enable, per the
existing file's own comment convention.
```

---

### Prompt 7 — A6: build the cross-dataset dedup scanner

```
STATUS.md's final entry (2026-07-08) recommends this as the first Phase 1
priority: "build the cross-dataset correlation/dedup scan used ad hoc this
session (for solar and electricity) into an actual reusable Phase 2 tool...
run it pairwise across every dataset pair in the final Phase 2 collection
before any zero-shot claim is made."

Read that STATUS.md entry and the electricity.md/solar_energy.md/
gift_eval.md/monash.md literature cards for exactly how those ad hoc checks
were done (sliding-window alignment, correlation threshold, byte-diff
check). Generalize that into scripts/analysis/dedup_scan.py:
- Takes any two standardized parquet files in this project's schema.
- Aligns series by whatever's feasible (matching frequency, sliding-window
  offset search like the electricity check did) and reports per-series-pair
  correlation, plus a byte/value-identical flag.
- Runs pairwise across all datasets currently in dataset/processed/
  (energy vs traffic vs weather vs multi_domain, all of it — the known
  duplicates so far are solar (3-way) and electricity (2-way suspected,
  confirmed this session per STATUS.md), but the whole point is to check
  pairs nobody's manually looked at yet, like the flagged-but-unchecked
  Traffic vs Monash's traffic_hourly).
- Writes results to a new dataset/manifests/dedup_scan_report.md with any
  pair above a correlation threshold (use 0.99 as a first pass, flag the
  threshold as configurable) called out explicitly.

Run it against the current dataset collection and report what it finds,
including confirming or refuting the Traffic-vs-traffic_hourly suspicion
STATUS.md left open.
```

---

### Prompt 8 — A7: read Meyer et al. and update the leakage matrix

```
Fetch and read arXiv:2510.13654 (Meyer et al. 2025, "Rethinking Evaluation
in the Era of Time Series Foundation Models: (Un)known Information Leakage
Challenges") in full — STATUS.md flags this as independently corroborating
and extending this project's own contamination findings (ETTh1/ETTh2/ETTm1
vs Lag-Llama and UniTime, the solar/GIFT-Eval/Monash lineage) and possibly
containing a template for the leakage matrix this project needs to build.

Do two things with it:
1. Update config/contamination_matrix.yaml with any model-dataset overrides
   this paper documents that aren't already captured (check what's already
   there first — lag-llama and unitime overrides for the ETT family are
   already seeded, per the existing file's comments).
2. Write a summary note at literature/notes/meyer_et_al_2025.md (new file,
   same card-style format as the other literature notes) covering: what
   matrix/methodology they use, which of our specific datasets and models
   they cover, and anything methodologically worth adopting into this
   project's own eventual Phase 2 leakage matrix design.

Don't invent citations or claims from this paper you can't actually verify
by reading it — if a claim from STATUS.md's summary of it can't be directly
confirmed in the text, flag that rather than restating it as fact.
```

---

## Notes on running these

- Each prompt assumes a fresh Codex session per task (cleaner diffs, easier
  to review, matches how the bridge phase's Claude sessions were logged one
  STATUS.md entry at a time). If your Codex setup keeps one long-running
  session instead, that's fine too, just paste them in order.
- None of these need GPU access. If Codex's environment for this repo has a
  GPU attached and idle, don't let it use it for these tasks — Bucket B in
  `NEXT_PHASE_TASKS.md` is separate and needs an explicit go-ahead in this
  conversation first.
- Have it append a `STATUS.md` entry at the end of every one of these, same
  as the bridge phase's convention — that log is what let this project's
  context survive across sessions and servers so far, don't break the habit
  now that there are two coding agents (Claude + Codex) potentially touching
  the same repo.