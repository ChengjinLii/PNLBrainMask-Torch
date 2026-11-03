import builtins
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pnl_brain_mask.inference import aggregate, normalize, orient, select_device
from pnl_brain_mask.model import CompNet, architecture
from pnl_brain_mask.weights import AUTHOR_FILES, check, load_model


def test_released_graph_structure():
    counts = Counter(layer["class_name"] for layer in architecture()["config"]["layers"])
    assert counts["Conv2D"] == 57
    assert counts["Conv2DTranspose"] == 15
    assert counts["BatchNormalization"] == 54
    assert len(architecture()["config"]["layers"]) == 177
    model = CompNet()
    assert model.output_names == ["final_op", "xfinal_op", "res_1_final_op"]


def test_no_tensorflow_or_h5py_needed_for_torch_inference(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in ("tensorflow", "keras", "conversion", "h5py"):
            raise AssertionError("Runtime imported offline-only dependency: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    torch.set_num_threads(1)
    model = CompNet()
    with torch.inference_mode():
        heads, captured = model(torch.zeros(1, 1, 256, 256), capture=("final_op",))
        pruned = model(torch.zeros(1, 1, 256, 256), segmentation_only=True)
    assert all(head.shape == (1, 1, 256, 256) for head in heads)
    assert torch.equal(pruned, captured["final_op"])


def test_training_and_wrong_shape_rejected():
    model = CompNet()
    with pytest.raises(ValueError, match="NCHW"):
        model(torch.zeros(1, 1, 32, 32))
    model.train()
    with pytest.raises(RuntimeError, match="inference only"):
        model(torch.zeros(1, 1, 256, 256))


@pytest.mark.parametrize("view", ["sagittal", "coronal", "axial"])
def test_orientation_is_exactly_invertible(view):
    data = np.arange(4 * 5 * 6).reshape(4, 5, 6)
    np.testing.assert_array_equal(orient(orient(data, view), view), data)


def test_threshold_and_float32_order():
    arrays = {view: np.array([0.45, 0.45001, 0.44999], dtype=np.float32)
              for view in ("sagittal", "coronal", "axial")}
    fused, mask = aggregate(arrays)
    np.testing.assert_array_equal(mask, [0, 1, 0])
    assert fused.dtype == np.float32


def test_normalization_and_invalid_data():
    data = np.linspace(-1, 100, 256**3, dtype=np.float32).reshape(256, 256, 256)
    output = normalize(data)
    expected = np.clip(data / np.float32(np.percentile(data, 99)), 0, 1)
    assert output.dtype == np.float32
    np.testing.assert_array_equal(output, expected)
    with pytest.raises(ValueError, match="positive"):
        normalize(np.zeros_like(data))
    data[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        normalize(data)


def test_no_silent_cuda_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device("auto")["resolved"] == "cpu"
    with pytest.raises(RuntimeError, match="CUDA requested"):
        select_device("cuda")


def test_hash_check_before_deserialization(tmp_path, monkeypatch):
    from pnl_brain_mask import weights
    receipt = {"format": 1, "architecture_sha256": weights.ARCHITECTURE_SHA256,
               "views": {"axial": {"file": "axial.pt", "source_sha256": AUTHOR_FILES["axial"][1], "sha256": "0" * 64}}}
    (tmp_path / "weights.json").write_text(json.dumps(receipt))
    (tmp_path / "axial.pt").write_bytes(b"untrusted")
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("Deserialized unchecked weights"))
    with pytest.raises(ValueError, match="SHA256"):
        load_model(tmp_path, "axial")


def test_converted_author_weights_if_present():
    root = Path(__file__).parents[1] / "weights"
    if not (root / "weights.json").is_file():
        pytest.skip("Download converted weights for this optional test")
    torch.set_num_threads(1)
    for view in AUTHOR_FILES:
        model = load_model(root, view)
        with torch.inference_mode():
            outputs = model(torch.zeros(1, 1, 256, 256))
        assert all(torch.isfinite(output).all() for output in outputs)
