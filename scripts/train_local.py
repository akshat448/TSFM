"""Local pretrain + probe + retrieval run against the real small CHISCO chunk
downloaded by download_chisco_chunk.sh (28 trials, subject 01, imagined
speech, run 045).

THIS IS A SMOKE TEST, NOT A RESULT. 28 trials is nowhere near enough data to
learn anything neurally meaningful; the point is to prove the pipeline runs
end-to-end on real (not synthetic) CHISCO data, and to establish the actual
chance-level baselines this small slice implies, per the corrected guidance
in eval_probe.py / __init__.py. Do not read the numbers this prints as
evidence of decoding ability -- read them as "did the code work."

Usage: python scripts/train_local.py [--ssl-epochs N] [--probe-epochs N]

Note on "epoch" here: each SSL/probe step below trains on the FULL training
split in one shot (no mini-batching -- 23 trials easily fits in memory/one
forward pass), so 1 step == 1 full epoch over the training data. That's why
--ssl-epochs directly controls the SSL_STEPS loop count below rather than
being a separate concept.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--ssl-epochs", type=int, default=15)
parser.add_argument("--probe-epochs", type=int, default=30)
cli_args = parser.parse_args()

from chisco_pipeline import (
    CHISCODataloader,
    collate_chisco,
    SpatialWhiteningFrontEnd,
    SincNetMambaEncoder,
    EMATeacher,
    HeteroscedasticPredictionHead,
    TeacherStudentDistillationLoss,
    LinearProbeEvaluator,
    TextAlignmentHead,
    FrozenTextEncoder,
    word_grouped_top10_retrieval,
)

torch.manual_seed(0)

PKL_PATH = REPO_ROOT / "data/derivatives/preprocessed_pkl/sub-01/eeg/sub-01_task-imagine_run-045_eeg.pkl"
if not PKL_PATH.exists():
    raise SystemExit(f"Missing {PKL_PATH} -- run scripts/download_chisco_chunk.sh first")

D_MODEL = 64
D_TEXT = 64
N_LAYERS = 2
D_STATE = 8
N_SINC_FILTERS = 8
SINC_KERNEL = 51
SSL_STEPS = cli_args.ssl_epochs
PROBE_STEPS = cli_args.probe_epochs
# Fixed independent of SSL_STEPS: the beta(t) anneal schedule describes how
# fast the precision-regularizing -log(Pi) term should ramp in, which is a
# property of the loss dynamics, not of how many epochs a given local run
# happens to use. Tying it to SSL_STEPS (previous bug) meant a short run
# rushed beta 0->1 in just a few steps, destabilizing the loss (observed:
# loss rose instead of fell over 3 epochs). 100 gives a gentle ramp that
# stays sane whether you run 3 epochs or 300.
ANNEAL_STEPS = 100

print("== Loading real CHISCO chunk ==")
ds = CHISCODataloader([PKL_PATH], sample_rate_hz=500.0, highpass_hz=0.5)
print(f"{len(ds)} trials loaded from {PKL_PATH.name}")

all_samples = [ds[i] for i in range(len(ds))]
n_val = max(4, len(all_samples) // 5)
train_samples, val_samples = all_samples[:-n_val], all_samples[-n_val:]
print(f"train={len(train_samples)} held-out={len(val_samples)}")

train_batch = collate_chisco(train_samples)
val_batch = collate_chisco(val_samples)
print("train eeg:", train_batch["eeg"].shape, "val eeg:", val_batch["eeg"].shape)

frontend = SpatialWhiteningFrontEnd(ds.cortical_indices(), ds.eog_indices())
encoder = SincNetMambaEncoder(
    n_cortical_channels=len(ds.cortical_indices()),
    sample_rate_hz=ds.sample_rate_hz,
    n_sinc_filters=N_SINC_FILTERS,
    sinc_kernel_size=SINC_KERNEL,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    d_state=D_STATE,
)
head = HeteroscedasticPredictionHead(D_MODEL)

print("\n== Self-supervised pretraining (teacher/student), real EEG ==")
ema = EMATeacher(encoder, momentum=0.9)
dist_loss = TeacherStudentDistillationLoss(anneal_steps=ANNEAL_STEPS)
opt = torch.optim.Adam(
    list(frontend.parameters()) + list(encoder.parameters()) + list(head.parameters()), lr=1e-3
)

train_cortical = frontend(train_batch["eeg"])
for step in range(SSL_STEPS):
    opt.zero_grad()
    student_tokens = encoder(train_cortical)
    mean, precision = head(student_tokens)
    with torch.no_grad():
        teacher_tokens = ema(train_cortical.detach())
    out = dist_loss(mean, precision, teacher_tokens, step=step)
    out["loss"].backward()
    opt.step()
    ema.update(encoder)
    print(
        f"step {step:2d}: loss={out['loss'].item():.4f} beta={out['beta']:.3f} "
        f"mean_precision={out['mean_precision'].item():.4f}"
    )
    # cortical activations used by the frontend graph get freed by backward();
    # recompute so both frontend and encoder keep training on the same
    # front-end parameters across steps rather than reusing a stale graph.
    train_cortical = frontend(train_batch["eeg"])

encoder.eval()
print("\n== Linear probe: held-out sentence-identity top-1 ==")
train_labels_str = train_batch["catalog_content"]
val_labels_str = val_batch["catalog_content"]
label_vocab = sorted(set(train_labels_str) | set(val_labels_str))
label_to_idx = {s: i for i, s in enumerate(label_vocab)}
n_classes = len(label_vocab)
chance_level = 1.0 / n_classes
print(f"{n_classes} distinct sentences in this slice -> chance top-1 = {chance_level:.3f}")

with torch.no_grad():
    train_cortical_probe = frontend(train_batch["eeg"])
    val_cortical_probe = frontend(val_batch["eeg"])

probe = LinearProbeEvaluator(encoder, d_model=D_MODEL, n_classes=n_classes)
probe_opt = torch.optim.Adam(probe.classifier.parameters(), lr=1e-2)
train_labels = torch.tensor([label_to_idx[s] for s in train_labels_str], dtype=torch.long)
val_labels = torch.tensor([label_to_idx[s] for s in val_labels_str], dtype=torch.long)

for step in range(PROBE_STEPS):
    probe_opt.zero_grad()
    logits = probe(train_cortical_probe)
    loss = torch.nn.functional.cross_entropy(logits, train_labels)
    loss.backward()
    probe_opt.step()
    if step % 10 == 0 or step == PROBE_STEPS - 1:
        print(f"probe step {step:2d}: train_loss={loss.item():.4f}")

held_out_acc = probe.top1_accuracy(val_cortical_probe, val_labels)
print(f"held-out top-1 accuracy: {held_out_acc:.3f} (chance = {chance_level:.3f})")

print("\n== Text alignment + word-grouped top-10 retrieval (untrained head) ==")
text_encoder = FrozenTextEncoder(d_text=D_TEXT, device="cpu")
align_head = TextAlignmentHead(d_model=D_MODEL, d_text=D_TEXT, n_query_tokens=2, n_heads=4)

with torch.no_grad():
    all_cortical = frontend(collate_chisco(all_samples)["eeg"])
    all_tokens = encoder(all_cortical)
    pooled_brain = align_head(all_tokens)
    text_embeds = text_encoder(collate_chisco(all_samples)["catalog_content"])

retrieval_acc = word_grouped_top10_retrieval(
    pooled_brain, text_embeds, collate_chisco(all_samples)["catalog_content"]
)
n_all = len(all_samples)
retrieval_chance = min(10, n_all - 1) / (n_all - 1)
print(
    f"word-grouped top-10 retrieval accuracy: {retrieval_acc:.3f} "
    f"(chance ~= {retrieval_chance:.3f} since all {n_all} sentences in this run are distinct, "
    "so this reduces to 'is a random top-10 out of the rest of the batch')"
)

print("\nDone. This ran entirely on real CHISCO data (not synthetic) on CPU.")
print("Numbers above are a pipeline smoke test, not a scientific result -- see module docstring.")
