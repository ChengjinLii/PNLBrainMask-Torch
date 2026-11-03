"""Compare the actual pinned upstream preprocessing/fusion/native cleanup code.

Only named scientific functions are loaded by AST; upstream startup, TensorFlow,
CLI, cleanup, and browser code are not executed. Inputs/outputs stay in --output.
"""

import argparse
import ast
import json
import os
from os import path
from pathlib import Path
import subprocess
import shutil
from types import SimpleNamespace

import nibabel as nib
import numpy as np

from pnl_brain_mask.inference import aggregate, normalize, orient
from pnl_brain_mask.pipeline import finish_native
from pnl_brain_mask.weights import check, sha256

SCRIPT_SHA256 = "de1180437804dcfdc6c4cdf697bed70c59ab8f5c196cc104be05519a1cb7d961"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--ants-bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    check(args.script, SCRIPT_SHA256)
    names = {"normalize", "split", "multi_view_fast", "ANTS_inverse_transform", "npy_to_nifti", "save_nifti"}
    tree = ast.parse(args.script.read_bytes())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in tree.body} != names:
        raise RuntimeError("Missing reviewed upstream function")
    namespace = {"np": np, "nib": nib, "path": path, "subprocess": subprocess,
                 "SUFFIX_NIFTI_GZ": "nii.gz", "SUFFIX_NIFTI": "nii", "SUFFIX_NHDR": "nhdr", "SUFFIX_NPY": "npy",
                 "args": SimpleNamespace(filter=None), "output_mask": []}
    exec(compile(tree, str(args.script), "exec"), namespace)
    registered = nib.load(args.prepared / "rigid-Warped.nii.gz")
    # Use a private copy: upstream normalize creates an adjacent output.
    registered_path = args.output / "registered.nii.gz"
    nib.save(registered, registered_path)
    upstream_normalized = namespace["normalize"](str(registered_path), 99)
    original = nib.load(upstream_normalized).get_fdata().astype(np.float32)
    actual = normalize(registered.get_fdata())
    np.testing.assert_array_equal(original, actual)
    probabilities = {}
    original_split = {}
    for view in ("sagittal", "coronal", "axial"):
        # Inputs mirror predict_mask's NHWC and upstream split() exactly.
        raw = np.load(args.paired / view / "volume/probability.npy")
        file = args.output / (view + "_tf.npy")
        np.save(file, raw[..., None])
        split = namespace["split"](str(file), [str(args.output / "baseline.nii.gz")], view=view)[0]
        original_split[view] = split
        probabilities[view] = orient(raw, view)
        np.testing.assert_array_equal(np.load(split)[..., 0], probabilities[view])
    fused_file = namespace["multi_view_fast"](*(original_split[view] for view in ("sagittal", "coronal", "axial")),
                                               str(args.output / "baseline.nii.gz"))
    _, mask = aggregate(probabilities)
    np.testing.assert_array_equal(np.load(fused_file), mask)
    template_tf = nib.load(args.paired / "tf_template_mask.nii.gz").get_fdata()
    template_torch = nib.load(args.paired / "torch_template_mask.nii.gz").get_fdata()
    os.environ["PATH"] = str(args.ants_bin) + os.pathsep + os.environ.get("PATH", "")
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "2"
    native = {}
    for name, template_mask in (("tf", template_tf), ("torch", template_torch)):
        folder = args.output / name
        folder.mkdir(exist_ok=True)
        target = folder / "native.nii.gz"
        finish_native(template_mask, registered, args.prepared / "baseline.nii.gz",
                      args.prepared / "rigid-0GenericAffine.mat", target, ants_bin=args.ants_bin, threads=2)
        native[name] = nib.load(target)
    # Independently execute author's original inverse-transform, fill and cast.
    private_baseline = args.output / "upstream_baseline.nii.gz"
    shutil.copyfile(args.prepared / "baseline.nii.gz", private_baseline)
    upstream_native = namespace["npy_to_nifti"]([str(registered_path)], [str(fused_file)],
        [str(private_baseline)], view="multi",
        reference=[str((args.prepared / "baseline.nii.gz").resolve())],
        omat=[str((args.prepared / "rigid-0GenericAffine.mat").resolve())])
    upstream_image = nib.load(upstream_native[0])
    a, b, upstream = [image.get_fdata() for image in (native["tf"], native["torch"], upstream_image)]
    np.testing.assert_array_equal(a, upstream)
    different = int(np.count_nonzero(a != b))
    dice = float(2 * np.count_nonzero((a > 0) & (b > 0)) / (np.count_nonzero(a) + np.count_nonzero(b)))
    geometry = all(image.shape == native["tf"].shape and np.allclose(image.affine, native["tf"].affine, rtol=0, atol=1e-4)
                   for image in native.values())
    result = {"upstream_script_sha256": sha256(args.script), "normalization_bitwise_equal": True,
              "view_orientation_bitwise_equal": True, "fusion_mask_bitwise_equal": True,
              "upstream_native_cleanup_bitwise_equal": True, "native_tf_torch_different_voxels": different,
              "native_tf_torch_dice": dice, "native_voxels": a.size, "geometry_equal": bool(geometry),
              "tf_brain_voxels": int(np.count_nonzero(a)), "torch_brain_voxels": int(np.count_nonzero(b))}
    (args.output / "contract.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
