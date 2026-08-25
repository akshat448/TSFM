"""Full-scale CHISCO pretrain + probe + retrieval training, meant to run on a
CUDA GPU (e.g. via slurm/train_rtx6000.sbatch), logging to Weights & Biases.

This is the "real" counterpart to scripts/train_local.py / train_read_imagine_probe.py:
those were deliberately tiny, CPU-only smoke tests on a handful of trials to
validate the pipeline and establish honest chance-level baselines (see their
docstrings). This script is what you actually train once you've decided the
architecture is worth scaling -- real minibatching, the CUDA `mamba_ssm`
kernel (falls back to the pure-PyTorch scan if unavailable), and full
experiment tracking instead of print statements.

Requires WANDB_API_KEY in the environment (see .env.sample -- `wandb.init`
picks it up automatically; no code-level API key handling needed).

Usage:
    python scripts/train_cluster.py \
        --data-dir data/derivatives/preprocessed_pkl \
        --task imagine --subjects 01,02,03,04,05 \
        --mamba-backend mamba_ssm \
        --wandb-project chisco-tsfm
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass  # WANDB_API_KEY must already be exported in the environment instead

import torch
import wandb
from torch.utils.data import DataLoader, random_split

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

CROP_SAMPLES_BY_TASK = {"imagine": 1651, "read": 2501}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default=str(REPO_ROOT / "data/derivatives/preprocessed_pkl"))
    p.add_argument("--task", type=str, default="imagine", choices=["imagine", "read"])
    p.add_argument("--subjects", type=str, default="01,02,03,04,05")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--ssl-epochs", type=int, default=100)
    p.add_argument("--probe-epochs", type=int, default=100)
    p.add_argument("--anneal-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--n-sinc-filters", type=int, default=32)
    p.add_argument("--sinc-kernel-size", type=int, default=101)
    p.add_argument("--mamba-backend", type=str, default="pure_pytorch", choices=["pure_pytorch", "mamba_ssm"])
    p.add_argument("--text-model-name", type=str, default="uer/t5-base-chinese-cluecorpussmall",
                    help="Frozen text encoder. Default is a real T5 checkpoint pretrained on Chinese "
                         "(CHISCO's sentences are Chinese -- google/flan-t5-base would mistokenize them).")
    p.add_argument("--d-text", type=int, default=768)
    p.add_argument("--n-query-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb-project", type=str, default="chisco-tsfm")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--checkpoint-dir", type=str, default=str(REPO_ROOT / "checkpoints"))
    return p


def discover_pkl_files(data_dir: str, task: str, subjects: list[str]) -> list[Path]:
    files = []
    for sub in subjects:
        pattern = os.path.join(data_dir, f"sub-{sub}", "eeg", f"sub-{sub}_task-{task}_run-*_eeg.pkl")
        files.extend(sorted(Path(p) for p in glob.glob(pattern)))
    return files


def main() -> None:
    args = build_arg_parser().parse_args()
    torch.manual_seed(args.seed)
    subjects = args.subjects.split(",")

    pkl_files = discover_pkl_files(args.data_dir, args.task, subjects)
    if not pkl_files:
        raise SystemExit(
            f"No pkl files found under {args.data_dir} for task={args.task}, subjects={subjects}. "
            "Run scripts/download_chisco_full.sh first."
        )
    print(f"Found {len(pkl_files)} run files: {[p.name for p in pkl_files]}")

    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name, config=vars(args))

    crop_samples = CROP_SAMPLES_BY_TASK[args.task]
    dataset = CHISCODataloader(pkl_files, sample_rate_hz=500.0, highpass_hz=0.5, crop_samples=crop_samples)
    print(f"{len(dataset)} total trials loaded")

    n_val = max(1, int(len(dataset) * args.val_frac))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"train={n_train} val={n_val}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_chisco)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_chisco)

    device = torch.device(args.device)
    frontend = SpatialWhiteningFrontEnd(dataset.cortical_indices(), dataset.eog_indices()).to(device)
    encoder = SincNetMambaEncoder(
        n_cortical_channels=len(dataset.cortical_indices()),
        sample_rate_hz=dataset.sample_rate_hz,
        n_sinc_filters=args.n_sinc_filters,
        sinc_kernel_size=args.sinc_kernel_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        mamba_backend=args.mamba_backend,
    ).to(device)
    head = HeteroscedasticPredictionHead(args.d_model).to(device)

    ema = EMATeacher(encoder, momentum=0.996).to(device)
    dist_loss = TeacherStudentDistillationLoss(anneal_steps=args.anneal_steps)
    opt = torch.optim.Adam(
        list(frontend.parameters()) + list(encoder.parameters()) + list(head.parameters()), lr=args.lr
    )

    print(f"\n== SSL pretraining, {args.ssl_epochs} epochs, backend={args.mamba_backend}, device={device} ==")
    global_step = 0
    for epoch in range(args.ssl_epochs):
        for batch in train_loader:
            eeg = batch["eeg"].to(device)
            opt.zero_grad()
            cortical = frontend(eeg)
            student_tokens = encoder(cortical)
            mean, precision = head(student_tokens)
            with torch.no_grad():
                teacher_tokens = ema(cortical.detach())
            out = dist_loss(mean, precision, teacher_tokens, step=global_step)
            out["loss"].backward()
            opt.step()
            ema.update(encoder)
            wandb.log(
                {
                    "ssl/loss": out["loss"].item(),
                    "ssl/beta": out["beta"],
                    "ssl/mean_precision": out["mean_precision"].item(),
                    "ssl/epoch": epoch,
                },
                step=global_step,
            )
            global_step += 1
        print(f"epoch {epoch}: last batch loss={out['loss'].item():.4f}")

    encoder.eval()

    print("\n== Linear probe: held-out sentence-identity top-1 ==")
    # Read text directly off dataset.trials rather than materializing full
    # CHISCOSample objects (which would re-run the highpass filter over every
    # trial's EEG just to read a label -- wasteful at full-dataset scale).
    all_texts_train = [dataset.trials[i]["text"].strip() for i in train_set.indices]
    all_texts_val = [dataset.trials[i]["text"].strip() for i in val_set.indices]
    label_vocab = sorted(set(all_texts_train) | set(all_texts_val))
    label_to_idx = {t: i for i, t in enumerate(label_vocab)}
    n_classes = len(label_vocab)
    chance_level = 1.0 / n_classes
    print(f"{n_classes} distinct sentences -> chance top-1 = {chance_level:.4f}")
    wandb.summary["probe/n_classes"] = n_classes
    wandb.summary["probe/chance_level"] = chance_level

    probe = LinearProbeEvaluator(encoder, d_model=args.d_model, n_classes=n_classes).to(device)
    probe_opt = torch.optim.Adam(probe.classifier.parameters(), lr=1e-2, weight_decay=1e-3)

    def eval_probe_top1() -> float:
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                eeg = batch["eeg"].to(device)
                labels = torch.tensor(
                    [label_to_idx[t] for t in batch["catalog_content"]], dtype=torch.long, device=device
                )
                cortical = frontend(eeg)
                logits = probe(cortical)
                correct += (logits.argmax(dim=-1) == labels).sum().item()
                total += labels.numel()
        return correct / max(total, 1)

    best_probe_acc = 0.0
    for step in range(args.probe_epochs):
        for batch in train_loader:
            eeg = batch["eeg"].to(device)
            labels = torch.tensor(
                [label_to_idx[t] for t in batch["catalog_content"]], dtype=torch.long, device=device
            )
            probe_opt.zero_grad()
            with torch.no_grad():
                cortical = frontend(eeg)
            logits = probe(cortical)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            probe_opt.step()
        acc = eval_probe_top1()
        best_probe_acc = max(best_probe_acc, acc)
        wandb.log({"probe/train_loss": loss.item(), "probe/held_out_top1": acc, "probe/epoch": step})
        if step % 10 == 0 or step == args.probe_epochs - 1:
            print(f"probe epoch {step}: loss={loss.item():.4f} held_out_top1={acc:.4f}")
    wandb.summary["probe/best_held_out_top1"] = best_probe_acc
    wandb.summary["probe/beat_chance"] = best_probe_acc > chance_level

    print("\n== Text alignment: word-grouped top-10 retrieval ==")
    text_encoder = FrozenTextEncoder(model_name=args.text_model_name, d_text=args.d_text, device=str(device))
    align_head = TextAlignmentHead(
        d_model=args.d_model, d_text=text_encoder.d_text, n_query_tokens=args.n_query_tokens
    ).to(device)
    align_opt = torch.optim.Adam(align_head.parameters(), lr=1e-3)

    def retrieval_pass(loader) -> float:
        pooled_list, text_list, content_list = [], [], []
        with torch.no_grad():
            for batch in loader:
                eeg = batch["eeg"].to(device)
                cortical = frontend(eeg)
                tokens = encoder(cortical)
                pooled = align_head(tokens)
                texts = text_encoder(batch["catalog_content"])
                pooled_list.append(pooled)
                text_list.append(texts)
                content_list.extend(batch["catalog_content"])
        pooled_all = torch.cat(pooled_list, dim=0)
        text_all = torch.cat(text_list, dim=0)
        return word_grouped_top10_retrieval(pooled_all, text_all, content_list)

    # brief supervised alignment training (contrastive-style via retrieval
    # metric proxy: maximize cosine sim to matching text, minimize to others)
    for step in range(50):
        for batch in train_loader:
            eeg = batch["eeg"].to(device)
            with torch.no_grad():
                cortical = frontend(eeg)
                tokens = encoder(cortical)
            align_opt.zero_grad()
            pooled = align_head(tokens)
            with torch.no_grad():
                texts = text_encoder(batch["catalog_content"])
            sims = pooled @ texts.T
            targets = torch.arange(sims.shape[0], device=device)
            loss = torch.nn.functional.cross_entropy(sims * 10.0, targets)
            loss.backward()
            align_opt.step()
        retrieval_acc = retrieval_pass(val_loader)
        wandb.log({"retrieval/train_loss": loss.item(), "retrieval/held_out_top10": retrieval_acc, "retrieval/epoch": step})
        if step % 10 == 0:
            print(f"align epoch {step}: loss={loss.item():.4f} held_out_top10={retrieval_acc:.4f}")

    final_retrieval = retrieval_pass(val_loader)
    wandb.summary["retrieval/final_held_out_top10"] = final_retrieval
    print(f"\nFinal held-out word-grouped top-10 retrieval: {final_retrieval:.4f}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "frontend": frontend.state_dict(),
            "encoder": encoder.state_dict(),
            "probe": probe.state_dict(),
            "align_head": align_head.state_dict(),
            "args": vars(args),
        },
        ckpt_dir / f"{run.id}.pt",
    )
    print(f"Saved checkpoint to {ckpt_dir / f'{run.id}.pt'}")
    wandb.finish()


if __name__ == "__main__":
    main()
