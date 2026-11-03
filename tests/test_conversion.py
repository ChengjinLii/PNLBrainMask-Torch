"""Optional actual-author tensor audit, independent of framework forward checks."""

import os
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from pnl_brain_mask.weights import AUTHOR_FILES, check, load_model


@pytest.mark.parametrize("view", list(AUTHOR_FILES))
def test_every_converted_weight_matches_author_tensor(view):
    if not os.environ.get("PNL_AUTHOR_MODELS"):
        pytest.skip("Set PNL_AUTHOR_MODELS for exhaustive real-HDF5 weight audit")
    source = Path(os.environ["PNL_AUTHOR_MODELS"])
    filename, digest = AUTHOR_FILES[view]
    check(source / filename, digest)
    converted = Path(__file__).parents[1] / "weights"
    model = load_model(converted, view)
    state = model.state_dict()
    checked = 0
    with h5py.File(source / filename) as f:
        for layer in model.spec:
            name, kind = layer["name"], layer["class_name"]
            if kind in ("Conv2D", "Conv2DTranspose"):
                np.testing.assert_array_equal(state[f"ops.{name}.weight"].numpy(),
                                              np.asarray(f[name][name]["kernel:0"]).transpose(3, 2, 0, 1))
                np.testing.assert_array_equal(state[f"ops.{name}.bias"].numpy(), np.asarray(f[name][name]["bias:0"]))
                checked += 2
            elif kind == "BatchNormalization":
                for original, target in (("gamma", "weight"), ("beta", "bias"),
                                         ("moving_mean", "running_mean"), ("moving_variance", "running_var")):
                    np.testing.assert_array_equal(state[f"ops.{name}.{target}"].numpy(), np.asarray(f[name][name][original + ":0"]))
                    checked += 1
                assert state[f"ops.{name}.num_batches_tracked"] == 0
    assert checked == 360  # 72 kernels+biases and 54 four-tensor BN groups.
