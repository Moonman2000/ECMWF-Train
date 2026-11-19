"""Start a training from the weights of a pretrained run, also when the new model has more datasets.

In anemoi-training 0.9, `training.fork_run_id` resumes the pretrained run: it loads the whole checkpoint strictly
into the new model, which fails as soon as the new model has a dataset the pretrained one did not have
("Missing key(s) in state_dict: ... obs ..."). This callback does the transfer learning instead:

  - every weight of the pretrained model that exists in the new model with the same name and shape is loaded
    (here: the encoder and decoder of `data`, the processor and the trainable graph parameters)
  - the other weights keep their random initial values (here: the encoder and decoder of `obs`)
  - the training itself starts afresh: step 0, new optimiser, the learning rate schedule of the new config

It runs in `setup`, before the optimiser is created. Freezing is set as usual with training.submodules_to_freeze.

In the config:

    diagnostics:
      callbacks:
      - _target_: pretrained_weights.LoadPretrainedWeights
        checkpoint: <training checkpoint (last.ckpt) of the pretrained run>

The folder of this file must be on the PYTHONPATH (helpers.run_training does that).
"""

import logging
from pathlib import Path

import pytorch_lightning as pl
import torch

LOGGER = logging.getLogger(__name__)


class LoadPretrainedWeights(pl.callbacks.Callback):
    """Load the weights of a pretrained model that fit the new model (same name and shape)."""

    def __init__(self, config, checkpoint: str) -> None:  # anemoi passes the training config first
        super().__init__()
        del config
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {self.checkpoint}")

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        del trainer, stage
        pretrained = torch.load(self.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
        model = pl_module.state_dict()

        fits = {k: v for k, v in pretrained.items() if k in model and model[k].shape == v.shape}
        wrong_shape = sorted(k for k in pretrained if k in model and model[k].shape != pretrained[k].shape)
        new = sorted(k for k in model if k not in pretrained)

        pl_module.load_state_dict(fits, strict=False)

        def count(keys):
            return sum(model[k].numel() for k in keys)

        LOGGER.info("Pretrained weights from %s", self.checkpoint)
        LOGGER.info("  loaded: %d tensors (%d parameters)", len(fits), count(fits))
        LOGGER.info("  new, randomly initialised: %d tensors (%d parameters), e.g. %s", len(new), count(new), new[:3])
        if wrong_shape:
            LOGGER.warning("  not loaded, different shape: %s", wrong_shape)
        trainable = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in pl_module.parameters() if not p.requires_grad)
        LOGGER.info("  trainable parameters: %d, frozen: %d", trainable, frozen)
