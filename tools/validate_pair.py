"""Reproducible TF/Torch paired validation; never needs plugin imports.

Use a separate --tf-python so TensorFlow is not installed in the Torch runtime.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import nibabel as nib
import numpy as np
import torch

from pnl_brain_mask.inference import VIEWS, aggregate, normalize, orient
from pnl_brain_mask.weights import AUTHOR_FILES, check, load_model, sha256

CAPTURE = ["conv2d_1", "batch_normalization_1", "max_pooling2d_1", "batch_normalization_12",
           "conv2d_transpose_1", "up6", "batch_normalization_22", "final_op", "xfinal_op",
           "res_a", "batch_normalization_44", "conv2d_transpose_15", "res_1_final_op"]
HEADS = {"final_op", "xfinal_op", "res_1_final_op"}


def metrics(reference, actual, *, head=False):
    difference = np.abs(np.asarray(reference) - np.asarray(actual))
    atol, rtol = (1e-5, 1e-4) if head else (1e-3, 1e-4)
    if reference.dtype == np.float64:
        atol, rtol = 1e-9, 1e-9
    finite = bool(np.isfinite(reference).all() and np.isfinite(actual).all())
    return {"max_abs": float(difference.max()), "mean_abs": float(difference.mean()),
            "rmse": float(np.sqrt(np.mean(np.square(difference, dtype=np.float64)))),
            "atol": atol, "rtol": rtol,
            "passed": finite and bool(np.allclose(reference, actual, atol=atol, rtol=rtol))}


def mask_metrics(a, b):
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    denominator = int(a.sum() + b.sum())
    return {"different_voxels": int(np.count_nonzero(a != b)), "voxels": a.size,
            "dice": float(2 * np.count_nonzero(a & b) / denominator) if denominator else 1.0}


def reference(args, input_path, output, view, *, volume=False):
    command = [str(args.tf_python), str(Path(__file__).with_name("tf_reference.py")),
               "--source", str(args.source), "--input", str(input_path), "--output", str(output), "--view", view]
    command += ["--precision", args.precision]
    command += ["--volume"] if volume else ["--capture", *CAPTURE]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    subprocess.run(command, env=env, check=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tf-python", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--volume", type=Path, required=True, help="Registered 256^3 NIfTI, not native DWI")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-volume", action="store_true")
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    for name, digest in AUTHOR_FILES.values():
        check(args.source / name, digest)
    volume = normalize(nib.load(args.volume).get_fdata())
    report = {"schema": 1, "torch": torch.__version__, "torch_device": "cpu", "threads": 2,
              "input_sha256": sha256(args.volume), "input_kind": "registered_b0",
              "cases": {}, "full_volume": {}, "retrained": False, "precision": args.precision}
    all_passed = True
    tf_probabilities, torch_probabilities = {}, {}
    for view in VIEWS:
        slices = orient(volume, view)
        if args.precision == "float64":
            slices = slices.astype(np.float64)
        # Zero, seeded random, and three anatomical slices from a real volume.
        cases = {"zero": np.zeros((1, 256, 256), dtype=np.float32),
                 "random_seed_2026": np.random.default_rng(2026).random((1, 256, 256), dtype=np.float32),
                 "anatomical_64_128_192": np.ascontiguousarray(slices[[64, 128, 192]])}
        model = load_model(args.model_dir, view)
        if args.precision == "float64":
            model = model.double()
            cases = {name: x.astype(np.float64) for name, x in cases.items()}
        for case, x in cases.items():
            folder = args.output / view / case
            folder.mkdir(parents=True, exist_ok=True)
            input_path = folder / "input.npy"
            np.save(input_path, x)
            reference(args, input_path, folder, view)
            _, tensors = model(torch.from_numpy(x[:, None]), capture=CAPTURE)
            with np.load(folder / "tensors.npz") as original:
                measured = {name: metrics(original[name], tensors[name].permute(0, 2, 3, 1).numpy(), head=name in HEADS)
                            for name in CAPTURE}
                measured["segmentation_mask"] = mask_metrics(original["final_op"] > 0.45,
                                                               tensors["final_op"].permute(0, 2, 3, 1).numpy() > 0.45)
            report["cases"][view + "/" + case] = measured
            all_passed &= all(m["passed"] for name, m in measured.items() if name != "segmentation_mask")
            pruned = model(torch.from_numpy(x[:, None]), segmentation_only=True)
            if not torch.equal(pruned, tensors["final_op"]):
                raise AssertionError("Segmentation graph pruning changed final_op")
            print(view, case, "final_op", measured["final_op"], flush=True)
            del tensors
        if args.full_volume:
            folder = args.output / view / "volume"
            folder.mkdir(parents=True, exist_ok=True)
            input_path = folder / "input.npy"
            np.save(input_path, slices)
            reference(args, input_path, folder, view, volume=True)
            output = np.empty(volume.shape, dtype=args.precision)
            for index in range(256):
                output[index] = model(torch.from_numpy(np.ascontiguousarray(slices[index:index + 1, None])),
                                      segmentation_only=True)[0, 0].numpy()
                if index % 32 == 0:
                    print(f"Torch {view}: {index}/256", flush=True)
            np.save(folder / "torch_probability.npy", output)
            original = np.load(folder / "probability.npy", mmap_mode="r")
            measured = metrics(original, output, head=True)
            report["full_volume"][view] = measured
            all_passed &= measured["passed"]
            tf_probabilities[view] = orient(original, view)
            torch_probabilities[view] = orient(output, view)
        report["tensorflow_runtime"] = json.loads((args.output / view / "zero/runtime.json").read_text())
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        del model
    if args.full_volume:
        tf_fused, tf_mask = aggregate(tf_probabilities)
        torch_fused, torch_mask = aggregate(torch_probabilities)
        report["fused_probability"] = metrics(tf_fused, torch_fused, head=True)
        report["template_mask"] = mask_metrics(tf_mask, torch_mask)
        all_passed &= report["fused_probability"]["passed"] and report["template_mask"]["dice"] >= 0.99999
        image = nib.load(args.volume)
        for name, mask in (("tf", tf_mask), ("torch", torch_mask)):
            nib.save(nib.Nifti1Image(mask, image.affine, image.header), args.output / (name + "_template_mask.nii.gz"))
    report["passed"] = bool(all_passed)
    report["wall_seconds"] = time.monotonic() - started
    report["limitations"] = ["CPU pairing only; CUDA parity not established", "Numerical agreement is not anatomical accuracy",
                             "No claim of multi-subject clinical validation"]
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not all_passed:
        raise SystemExit("Paired validation failed; inspect report.json")


if __name__ == "__main__":
    main()
