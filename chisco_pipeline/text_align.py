"""Text alignment: frozen-LLM text embeddings + learned-query cross-attention
pooling of brain latents, evaluated by word-grouped top-10 retrieval.

ZERO-FUSION DESIGN (plan correction #7 / user's explicit choice): the two
modalities are encoded completely independently.
  - Text embeddings come only from a frozen T5/BART encoder over
    `catalog_content` (see dataloader.py) -- the brain encoder never
    participates in producing them.
  - Brain embeddings come only from a fixed set of LEARNABLE QUERY VECTORS
    that cross-attend over the (already fully-trained, frozen at this stage)
    brain encoder's token sequence. The queries are parameters of this
    module; text embeddings are never used as a query, key, or value in any
    brain-side computation, not even post-hoc. This is the strict reading of
    "zero fusion during encoding" -- the text embedding only re-enters the
    picture afterward, purely to define the retrieval loss/metric that
    aligns the two independently-produced vector spaces.

This also means the module works even if you never see the text encoder
during evaluation of the brain side: `pool_brain_latents` alone is a valid
"encode a trial" operation with no text dependency at all.
"""

from __future__ import annotations

import torch
from torch import nn

try:
    from transformers import AutoModel, AutoTokenizer

    _HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover - environment-dependent
    _HAS_TRANSFORMERS = False


class FrozenTextEncoder(nn.Module):
    """Wraps a frozen pretrained T5/BART encoder producing a single pooled
    embedding per RSVP `catalog_content` string.

    Shapes: list[str] of length B -> (B, d_text)

    Falls back to a fixed random-projection bag-of-hashes encoder if
    `transformers` isn't installed, purely so the rest of the pipeline (shape
    contracts, retrieval metric) can be smoke-tested without the dependency.
    This fallback is NOT semantically meaningful and must not be used for any
    real evaluation -- it exists only to keep local shape/gradient tests
    runnable everywhere.
    """

    def __init__(self, model_name: str = "google/flan-t5-base", d_text: int | None = None, device: str = "cpu"):
        super().__init__()
        self.device = device
        if _HAS_TRANSFORMERS:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.d_text = self.model.config.d_model if hasattr(self.model.config, "d_model") else self.model.config.hidden_size
            self._stub = False
        else:  # pragma: no cover
            assert d_text is not None, "d_text required when transformers is unavailable"
            self.d_text = d_text
            self._stub = True
            self._stub_proj = nn.Embedding(50_000, d_text)  # frozen random hash embedding

    @torch.no_grad()
    def forward(self, texts: list[str]) -> torch.Tensor:
        """texts: list[str], len B -> (B, d_text), L2-normalized."""
        if not self._stub:
            enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            if hasattr(self.model, "get_encoder"):
                out = self.model.get_encoder()(**enc).last_hidden_state  # (B, L, d_text)
            else:
                out = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)  # (B, L, 1)
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)  # (B, d_text)
        else:  # pragma: no cover
            hashes = torch.tensor([abs(hash(t)) % 50_000 for t in texts], dtype=torch.long)
            pooled = self._stub_proj(hashes)
        return torch.nn.functional.normalize(pooled, dim=-1)


class TextAlignmentHead(nn.Module):
    """Learned-query cross-attention pooling of brain latent tokens into the
    frozen text embedding's dimensionality, plus a word-grouped top-10
    retrieval metric.

    Shapes
    ------
    brain_tokens: (B, T, d_model)     -- SincNetMambaEncoder output (frozen)
    pooled_brain: (B, d_text)          -- one vector per trial, L2-normalized
    text_embeds:  (B, d_text)          -- FrozenTextEncoder output, independent
    """

    def __init__(self, d_model: int, d_text: int, n_query_tokens: int = 4, n_heads: int = 8):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(n_query_tokens, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.out_proj = nn.Linear(d_model * n_query_tokens, d_text)

    def pool_brain_latents(self, brain_tokens: torch.Tensor) -> torch.Tensor:
        """brain_tokens: (B, T, d_model) -> (B, d_text), L2-normalized.

        The learned query tokens are the ONLY query into this attention op;
        brain_tokens serve as key/value. No text embedding is involved.
        """
        B = brain_tokens.shape[0]
        queries = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, n_query, d_model)
        pooled, _ = self.cross_attn(queries, brain_tokens, brain_tokens)  # (B, n_query, d_model)
        pooled = pooled.reshape(B, -1)  # (B, n_query * d_model)
        pooled = self.out_proj(pooled)  # (B, d_text)
        return torch.nn.functional.normalize(pooled, dim=-1)

    def forward(self, brain_tokens: torch.Tensor) -> torch.Tensor:
        return self.pool_brain_latents(brain_tokens)


@torch.no_grad()
def word_grouped_top10_retrieval(
    pooled_brain: torch.Tensor,
    text_embeds: torch.Tensor,
    catalog_content: list[str],
) -> float:
    """Word-grouped top-10 retrieval accuracy in the raw latent space -- no
    UMAP/t-SNE or any other dimensionality reduction, per spec Part 5.

    For each brain-side query vector, rank all trials in the batch by cosine
    similarity of their text embedding; count a hit if ANY of the top-10
    retrieved trials shares the same ground-truth `catalog_content` word/
    label as the query (word-grouped: multiple trials can share a label, and
    retrieving any of them counts).

    Shapes
    ------
    pooled_brain, text_embeds: (B, d_text), both L2-normalized
    catalog_content: list[str] of length B, the ground-truth RSVP text
        (matched against directly here; group by exact string equality --
        if CHISCO's real labels need coarser word-level grouping, e.g.
        stemming, do that upstream and pass the grouped label instead of the
        raw sentence).

    Returns: float in [0, 1], fraction of queries with >=1 correct hit in
    their top-10.
    """
    B = pooled_brain.shape[0]
    sims = pooled_brain @ text_embeds.T  # (B, B) cosine sim, since inputs are L2-normalized
    # Exclude the trivial self-pair (i, i): pooled_brain[i] and text_embeds[i]
    # come from the SAME trial, so without masking the diagonal every query
    # would retrieve its own exact match by construction and the metric would
    # be meaningless. We want "can it find another trial with the same
    # ground-truth label", not "does it recognize itself".
    sims.fill_diagonal_(float("-inf"))
    k = min(10, B - 1)
    topk = sims.topk(k, dim=-1).indices  # (B, k)

    hits = 0
    for i in range(B):
        retrieved_labels = {catalog_content[j] for j in topk[i].tolist()}
        if catalog_content[i] in retrieved_labels:
            hits += 1
    return hits / B
