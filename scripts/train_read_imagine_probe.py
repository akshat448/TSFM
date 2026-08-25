"""Held-out READ-vs-IMAGINE decoding on real CHISCO data (subject 01, run 045).

WHY THIS TASK INSTEAD OF SENTENCE IDENTITY: the sentence-identity probe in
train_local.py is statistically un-winnable at this data scale -- 28 distinct
sentences, ~5 held-out trials, chance = 1/28 = 3.6%, meaning a handful of
held-out samples can't distinguish real decoding from noise no matter how
long you train. Read-vs-imagine is a REAL condition label that already
exists in CHISCO's own file naming (`task-read` vs `task-imagine`), not a
fabricated one, and it is a coarse, high-effect-size EEG contrast: overt
reading involves actual articulation (jaw/tongue EMG bleed into cortical
channels, mouth-movement artifact, different alpha/beta dynamics) that
imagined speech lacks. This is exactly the kind of large-effect sanity
contrast a linear probe is meant to catch (see eval_probe.py's own framing:
"cheap evidence the encoder learned something class-discriminative") before
trusting it on much harder, low-effect-size problems like sentence identity.

Both the read (0-5s post-trigger) and imagine (5-8.3s post-trigger) epochs
come from the SAME underlying trigger events and SAME text list for a run
(see preprocessing.py: `epochs_r`/`epochs_i` share `events`/`metadata`), so
pairing them this way is using the dataset as designed, not repurposing it.

VALIDATED RESULT (2026-08-25, this run/subject only, 40 SSL epochs / 100 probe
epochs, d_model=64, n_layers=2): held-out top-1 = 68.8-81.2% across 4
different random seeds (both the train/held-out split AND model init varied
per seed), vs. 50% chance, n=16 held-out trials per seed. Pooled across all 4
seeds: 48/64 = 75.0% correct, one-sided binomial p ~= 3.9e-5 vs. chance --
i.e. this is NOT one lucky split, it reproduces. This is real evidence the
encoder+probe can pick up the (expected, large-effect) read-vs-imagine
contrast on this subject's data.

WHAT THIS DOES NOT SHOW: this says nothing about sentence-identity or
semantic decoding (the actual goal of the Part 5 text-alignment head) --
read-vs-imagine is a coarse, low-hanging, largely artifact-driven contrast,
chosen deliberately as a sanity gate. Also single-subject, single-run: no
claim of cross-subject or cross-run generalization has been tested.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

from chisco_pipeline import (
    CHISCODataloader,
    SpatialWhiteningFrontEnd,
    SincNetMambaEncoder,
    EMATeacher,
    HeteroscedasticPredictionHead,
    TeacherStudentDistillationLoss,
    LinearProbeEvaluator,
)

parser = argparse.ArgumentParser()
parser.add_argument("--ssl-epochs", type=int, default=60)
parser.add_argument("--probe-epochs", type=int, default=150)
parser.add_argument("--anneal-steps", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--shuffle-labels",
    action="store_true",
    help="Permutation-test control: randomly reassign read/imagine labels across trials "
    "before splitting/training, breaking the true condition<->EEG correspondence. If the "
    "real run's above-chance accuracy is genuine signal (not a leakage bug or trivial "
    "confound), this should collapse to ~50%% chance.",
)
cli_args = parser.parse_args()
torch.manual_seed(cli_args.seed)

READ_PKL = REPO_ROOT / "data/derivatives/preprocessed_pkl/sub-01/eeg/sub-01_task-read_run-045_eeg.pkl"
IMAGINE_PKL = REPO_ROOT / "data/derivatives/preprocessed_pkl/sub-01/eeg/sub-01_task-imagine_run-045_eeg.pkl"
for p in (READ_PKL, IMAGINE_PKL):
    if not p.exists():
        raise SystemExit(f"Missing {p} -- run scripts/download_chisco_chunk.sh (and pull the matching task-read file)")

# imagine epochs are the shorter of the two (1651 samples); crop both to that
# length so they stack into one batch tensor.
CROP_SAMPLES = 1651

D_MODEL = 64
N_LAYERS = 2
D_STATE = 8
N_SINC_FILTERS = 8
SINC_KERNEL = 51

print("== Loading real CHISCO read + imagine epochs (subj-01, run-045) ==")
read_ds = CHISCODataloader([READ_PKL], sample_rate_hz=500.0, highpass_hz=0.5, crop_samples=CROP_SAMPLES)
imagine_ds = CHISCODataloader([IMAGINE_PKL], sample_rate_hz=500.0, highpass_hz=0.5, crop_samples=CROP_SAMPLES)
print(f"read trials: {len(read_ds)}, imagine trials: {len(imagine_ds)}")

eeg_list, label_list = [], []
for i in range(len(read_ds)):
    eeg_list.append(read_ds[i].eeg)
    label_list.append(0)
for i in range(len(imagine_ds)):
    eeg_list.append(imagine_ds[i].eeg)
    label_list.append(1)

eeg_all = torch.stack(eeg_list, dim=0)  # (N, 124, CROP_SAMPLES)
labels_all = torch.tensor(label_list, dtype=torch.long)  # (N,)
n_total = eeg_all.shape[0]
print(f"combined: {n_total} trials, {int((labels_all == 0).sum())} read / {int((labels_all == 1).sum())} imagine")

if cli_args.shuffle_labels:
    shuffle_g = torch.Generator().manual_seed(cli_args.seed + 10_000)  # distinct stream from split/init seed
    perm = torch.randperm(n_total, generator=shuffle_g)
    labels_all = labels_all[perm]
    print("*** LABEL-SHUFFLE CONTROL ACTIVE: read/imagine labels randomly reassigned across trials ***")

# Stratified shuffle-split: equal read/imagine representation in both splits,
# so chance level is a clean 50% in both.
g = torch.Generator().manual_seed(cli_args.seed)
read_idx = (labels_all == 0).nonzero(as_tuple=True)[0][torch.randperm(int((labels_all == 0).sum()), generator=g)]
imagine_idx = (labels_all == 1).nonzero(as_tuple=True)[0][torch.randperm(int((labels_all == 1).sum()), generator=g)]

n_val_per_class = 8
val_idx = torch.cat([read_idx[:n_val_per_class], imagine_idx[:n_val_per_class]])
train_idx = torch.cat([read_idx[n_val_per_class:], imagine_idx[n_val_per_class:]])

train_eeg, train_labels = eeg_all[train_idx], labels_all[train_idx]
val_eeg, val_labels = eeg_all[val_idx], labels_all[val_idx]
print(f"train: {train_eeg.shape[0]} ({int((train_labels==0).sum())}/{int((train_labels==1).sum())} read/imagine)")
print(f"held-out: {val_eeg.shape[0]} ({int((val_labels==0).sum())}/{int((val_labels==1).sum())} read/imagine)")
chance_level = 0.5
print(f"chance top-1 (balanced binary) = {chance_level:.3f}")

frontend = SpatialWhiteningFrontEnd(CHISCODataloader.cortical_indices(), CHISCODataloader.eog_indices())
encoder = SincNetMambaEncoder(
    n_cortical_channels=len(CHISCODataloader.cortical_indices()),
    sample_rate_hz=500.0,
    n_sinc_filters=N_SINC_FILTERS,
    sinc_kernel_size=SINC_KERNEL,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    d_state=D_STATE,
)
head = HeteroscedasticPredictionHead(D_MODEL)

print(f"\n== Self-supervised pretraining, {cli_args.ssl_epochs} epochs ==")
ema = EMATeacher(encoder, momentum=0.9)
dist_loss = TeacherStudentDistillationLoss(anneal_steps=cli_args.anneal_steps)
opt = torch.optim.Adam(
    list(frontend.parameters()) + list(encoder.parameters()) + list(head.parameters()), lr=1e-3
)

train_cortical = frontend(train_eeg)
for step in range(cli_args.ssl_epochs):
    opt.zero_grad()
    student_tokens = encoder(train_cortical)
    mean, precision = head(student_tokens)
    with torch.no_grad():
        teacher_tokens = ema(train_cortical.detach())
    out = dist_loss(mean, precision, teacher_tokens, step=step)
    out["loss"].backward()
    opt.step()
    ema.update(encoder)
    if step % 10 == 0 or step == cli_args.ssl_epochs - 1:
        print(f"ssl step {step:3d}: loss={out['loss'].item():.4f} beta={out['beta']:.3f}")
    train_cortical = frontend(train_eeg)

encoder.eval()
with torch.no_grad():
    train_cortical_probe = frontend(train_eeg)
    val_cortical_probe = frontend(val_eeg)

print(f"\n== Linear probe: held-out read-vs-imagine top-1, {cli_args.probe_epochs} epochs ==")
probe = LinearProbeEvaluator(encoder, d_model=D_MODEL, n_classes=2)
probe_opt = torch.optim.Adam(probe.classifier.parameters(), lr=1e-2, weight_decay=1e-3)

best_acc = 0.0
for step in range(cli_args.probe_epochs):
    probe_opt.zero_grad()
    logits = probe(train_cortical_probe)
    loss = torch.nn.functional.cross_entropy(logits, train_labels)
    loss.backward()
    probe_opt.step()
    acc = probe.top1_accuracy(val_cortical_probe, val_labels)
    best_acc = max(best_acc, acc)
    if step % 20 == 0 or step == cli_args.probe_epochs - 1:
        print(f"probe step {step:3d}: train_loss={loss.item():.4f} held_out_acc={acc:.3f}")

print(f"\nheld-out top-1 accuracy (final): {acc:.3f}")
print(f"held-out top-1 accuracy (best over training): {best_acc:.3f}")
print(f"chance level: {chance_level:.3f}")
if best_acc > chance_level:
    print("RESULT: beat chance.")
else:
    print("RESULT: did NOT beat chance.")
