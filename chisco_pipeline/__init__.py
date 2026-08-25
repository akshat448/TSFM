"""CHISCO EEG -> semantic text embedding pipeline.

CHISCO is OpenNeuro ds005170 (https://openneuro.org/datasets/ds005170), an
IMAGINED-SPEECH dataset (not RSVP -- see dataloader.py for the full,
empirically-verified schema: real field names are "text"/"input_features",
122 cortical + VEO + HEO + Trigger channels at 500 Hz, no EMG channel).
A small real chunk (one run, 28 trials, subject 01) is checked into
data/derivatives/preprocessed_pkl/ via scripts/download_chisco_chunk.sh for
local smoke-testing; see scripts/train_local.py for a runnable pretrain +
probe + retrieval loop against it.

*** SCALE-UP GATE ***
Everything in this package is designed to be run first on this small local
slice (or a synthetic one), CPU/MPS or a single local GPU. The go/no-go
signal for moving to a bigger run (Modal, or the RTX 6000 you have access to)
is `LinearProbeEvaluator.top1_accuracy` (see eval_probe.py) clearing the
*actual* chance level for however many distinct sentences are in your current
label set (not a fixed percentage -- real CHISCO has no fixed word/category
vocabulary, see eval_probe.py for why). Only once that's cleared is it worth
paying for full-dataset training with the real CUDA `mamba-ssm` kernels;
scaling up before that gate just reproduces the same failure faster and more
expensively.

Pipeline order:
    CHISCODataloader
      -> SpatialWhiteningFrontEnd
      -> SincNetMambaEncoder            (self-supervised pretraining via
                                          TeacherStudentDistillationLoss)
      -> LinearProbeEvaluator           (held-out top-1 sanity gate, above)
      -> TextAlignmentHead              (word-grouped top-10 retrieval)
"""

from .dataloader import CHISCODataloader, CHISCOSample, collate_chisco
from .raw_dataloader import CHISCORawDataloader
from .frontend import SpatialWhiteningFrontEnd
from .encoder import SincNetMambaEncoder, SincConv1d, CausalMambaBlock, CausalSelectiveSSM
from .losses import TeacherStudentDistillationLoss, EMATeacher, HeteroscedasticPredictionHead
from .eval_probe import LinearProbeEvaluator
from .text_align import TextAlignmentHead, FrozenTextEncoder, word_grouped_top10_retrieval

__all__ = [
    "CHISCODataloader",
    "CHISCOSample",
    "collate_chisco",
    "CHISCORawDataloader",
    "SpatialWhiteningFrontEnd",
    "SincNetMambaEncoder",
    "SincConv1d",
    "CausalMambaBlock",
    "CausalSelectiveSSM",
    "TeacherStudentDistillationLoss",
    "EMATeacher",
    "HeteroscedasticPredictionHead",
    "LinearProbeEvaluator",
    "TextAlignmentHead",
    "FrozenTextEncoder",
    "word_grouped_top10_retrieval",
]
