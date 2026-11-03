"""Checked author assets and deterministic, TensorFlow-free weight conversion."""

import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import urllib.request

from .model import CompNet

SOURCE_COMMIT = "d7fe0fd4976c2d5a3489526f75d2a00dae3e1fe1"
SOURCE_URL = "https://github.com/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation"
MODEL_URL = SOURCE_URL + "/releases/download/v0.3/model_folder.tar.gz"
MODEL_SHA256 = "c67210fd59e5bcaf8e0e730bdde9d8f28b817148a74d9586a39c9b6e86df5191"
TEMPLATE_URL = "https://www.nitrc.org/frs/download.php/11290/IITmean_b0_256.nii.gz"
TEMPLATE_SHA256 = "2bb1a7bbe448702033a515e76164df6472828422ecef5cb4f996d91ec126dead"
ARCHITECTURE_SHA256 = "989fd5a64d9307268441c3525d59b31d59c20db864f0262ef53ad53ec495bdae"
AUTHOR_FILES = {
    "axial": ("weights-axial-improvement-08.h5", "e11d3c410769c68844109a86eed40ebedc655ea03c14f6f54f4f13749c5ead18"),
    "coronal": ("weights-coronal-improvement-08.h5", "248dcf90752d4e0a2e22a136c5f0ca6d3c9cb9b628d131ce869d9ef6cbea702c"),
    "sagittal": ("weights-sagittal-improvement-09.h5", "27aaafddac65617e0653c4da4fd0c650529fcc8150b4efd6b66135070ebc01bc"),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(path, expected):
    if sha256(path) != expected:
        raise ValueError("SHA256 mismatch: " + str(path))


def download(url, destination, expected):
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError("Download destination cannot be a symlink")
    if destination.is_file() and sha256(destination) == expected:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                size = 0
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    size += len(block)
                    if size > 512 * 1024**2:
                        raise ValueError("Download exceeds size limit")
                    stream.write(block)
            stream.close()
            check(temporary, expected)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def acquire_author_weights(destination, *, accept_terms=False):
    if not accept_terms:
        raise PermissionError("Review PNL software/data terms before downloading: " + SOURCE_URL)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive = download(MODEL_URL, destination / "model_folder.tar.gz", MODEL_SHA256)
    wanted = dict(AUTHOR_FILES.values())
    wanted["CompNetBasicModel.json"] = ARCHITECTURE_SHA256
    with tarfile.open(archive, "r:gz") as stream:
        for name, digest in wanted.items():
            matches = [m for m in stream.getmembers() if Path(m.name).name == name]
            if len(matches) != 1 or not matches[0].isfile() or matches[0].size > 256 * 1024**2:
                raise ValueError("Unexpected author archive contents: " + name)
            target = destination / name
            if target.is_symlink():
                raise ValueError("Asset cannot be a symlink")
            with stream.extractfile(matches[0]) as source, target.open("wb") as output:
                import shutil
                shutil.copyfileobj(source, output)
            check(target, digest)
    return destination


def convert(source, destination):
    import h5py
    import numpy as np
    import torch

    source, destination = Path(source), Path(destination)
    for name, digest in AUTHOR_FILES.values():
        check(source / name, digest)
    check(source / "CompNetBasicModel.json", ARCHITECTURE_SHA256)
    destination.mkdir(parents=True, exist_ok=True)
    receipt = {"format": 1, "source_commit": SOURCE_COMMIT, "source_url": MODEL_URL,
               "archive_sha256": MODEL_SHA256, "architecture_sha256": ARCHITECTURE_SHA256,
               "conversion": "HWIO/HWOI -> OIHW/IOHW; BN gamma/beta/mean/variance unchanged",
               "retrained": False, "views": {}}
    for view, (name, digest) in AUTHOR_FILES.items():
        model = CompNet()
        state = model.state_dict()
        with h5py.File(source / name, "r") as weights:
            for layer in model.spec:
                layer_name, kind = layer["name"], layer["class_name"]
                if kind in ("Conv2D", "Conv2DTranspose"):
                    group = weights[layer_name][layer_name]
                    # HWOI for transpose -> IOHW, no spatial flip.
                    array = np.asarray(group["kernel:0"]).transpose(3, 2, 0, 1).copy()
                    state[f"ops.{layer_name}.weight"] = torch.from_numpy(array)
                    state[f"ops.{layer_name}.bias"] = torch.from_numpy(np.asarray(group["bias:0"]))
                elif kind == "BatchNormalization":
                    group = weights[layer_name][layer_name]
                    for original, target in (("gamma", "weight"), ("beta", "bias"),
                                             ("moving_mean", "running_mean"), ("moving_variance", "running_var")):
                        state[f"ops.{layer_name}.{target}"] = torch.from_numpy(np.asarray(group[original + ":0"]))
        model.load_state_dict(state, strict=True)
        target = destination / f"{view}.pt"
        torch.save(state, target)
        receipt["views"][view] = {"file": target.name, "sha256": sha256(target),
                                    "source_file": name, "source_sha256": digest,
                                    "state_tensors": len(state),
                                    "parameters": sum(p.numel() for p in model.parameters())}
    (destination / "weights.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def load_model(directory, view, device="cpu"):
    import torch

    directory = Path(directory)
    receipt = json.loads((directory / "weights.json").read_text())
    if receipt["format"] != 1 or receipt["architecture_sha256"] != ARCHITECTURE_SHA256:
        raise ValueError("Unrecognized converted model")
    record = receipt["views"][view]
    if record["file"] != f"{view}.pt" or record["source_sha256"] != AUTHOR_FILES[view][1]:
        raise ValueError("Unexpected model provenance")
    path = directory / f"{view}.pt"
    check(path, record["sha256"])
    model = CompNet()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()
