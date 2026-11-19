"""An imputer for gridded observations as a TARGET.

In anemoi a dataset must be free of NaNs after its pre-processors (checked on the first batch), so the
gaps of a gridded observation dataset (grid points without an observation) have to be filled in, e.g.
with the mean. The filled-in values must then be left out of the loss. anemoi's InputImputer does that
with a loss mask, the `nan_mask_weights` scaler, but takes the mask from the FIRST time step of the
sample (an input time). That is right for gaps that do not change in time (e.g. the land of an SST
field), but observations change every time step: the loss needs the gaps of the TARGET time.

ObsImputer
  - fills the NaNs with the mean, like InputImputer
  - sets the loss mask from the gaps of the last time step of the sample (the target, with rollout 1)
  - in training and validation, puts NaNs back at the gaps of the target time after the model (so that
    validation metrics with ignore_nans: True only count real observations)
  - in inference (AnemoiModelInterface.predict_step), does not put NaNs back: the model predicts everywhere

Works with anemoi-models 0.12 (the version of this course) and with later versions.

With the richer batches of anemoi-core (coming soon), observations are tabular: a sample holds only the
observations there are, so no filling-in and no mask are needed.
"""

import torch
from anemoi.models.preprocessing.imputer import InputImputer


class ObsImputer(InputImputer):
    """InputImputer with the gaps of the target time for the loss mask and the output."""

    def transform(self, x: torch.Tensor, in_place: bool = True, **kwargs) -> torch.Tensor:
        # A training/validation sample has all variables (inference inputs have no diagnostic variables)
        # and all time steps: the inputs and the target.
        self._training_sample = x.shape[-1] == self.num_training_input_vars and not kwargs.get("skip_imputation")
        if self._training_sample:
            nan_locations = self.get_nans(x)  # (batch, time, grid, variable), before filling in
        x = super().transform(x, in_place=in_place, **kwargs)
        if self._training_sample:
            # super() took the gaps of time step 0 (an input time): use the last time step (the target)
            self.target_nan_locations = nan_locations[:, -1]  # (batch, grid, variable)
            for idx_src, idx_dst in zip(self.index_training_input, self.index_inference_output):
                if idx_src is not None and idx_dst is not None:
                    self.loss_mask_training[..., idx_dst] = (~nan_locations[:, -1, ..., idx_src]).int()
        return x

    def inverse_transform(self, x: torch.Tensor, in_place: bool = True, **kwargs) -> torch.Tensor:
        if not getattr(self, "_training_sample", False):
            return x  # inference: keep the prediction everywhere
        if not in_place:
            x = x.clone()
        index = self.index_training_output if x.shape[-1] == self.num_training_output_vars else self.index_inference_output
        for idx_src, idx_dst in zip(self.index_training_input, index):
            if idx_src is not None and idx_dst is not None:
                x[..., idx_dst][self._expand_subset_mask(x, idx_src, self.target_nan_locations)] = torch.nan
        return x
