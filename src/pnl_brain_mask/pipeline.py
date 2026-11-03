"""Modified, shell-free PNL native-grid pipeline with unchanged ANTs operations.

Original source: pnlbwh/CNN-Diffusion-MRIBrain-Segmentation, pinned in weights.py.
No implicit anatomical reorientation, optional morphology, or retraining.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import nibabel as nib
import numpy as np
import torch

from . import __version__
from .inference import normalize, predict, select_device
from .weights import SOURCE_COMMIT, TEMPLATE_SHA256, check, sha256


def run_ants(arguments, *, ants_bin=None, threads=1):
    env = os.environ.copy()
    if ants_bin:
        env["PATH"] = str(ants_bin) + os.pathsep + env.get("PATH", "")
        env["ANTSPATH"] = str(ants_bin)
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
    executable = shutil.which(arguments[0], path=env.get("PATH"))
    if executable is None:
        raise FileNotFoundError("Required ANTs command: " + arguments[0])
    if not ants_bin and arguments[0] == "antsRegistrationSyNQuick.sh":
        env["ANTSPATH"] = str(Path(executable).absolute().parent)
    subprocess.run([executable, *map(str, arguments[1:])], env=env, check=True)


def ants_workspace(work, template, ants_bin=None):
    """Give the ANTs shell script whitespace-free tool and template aliases."""
    if any(char.isspace() for char in str(work)):
        raise ValueError("ANTs work_dir must not contain whitespace")
    binary = work / "ants-bin"
    binary.mkdir()
    search = os.pathsep.join(filter(None, (str(ants_bin) if ants_bin else "", os.environ.get("PATH", ""))))
    for command in ("antsRegistrationSyNQuick.sh", "antsRegistration", "antsApplyTransforms", "ImageMath", "PrintHeader"):
        executable = shutil.which(command, path=search)
        if executable is None:
            raise FileNotFoundError("Required ANTs command: " + command)
        (binary / command).symlink_to(Path(executable).absolute())
    reference = work / "template.nii.gz"
    reference.symlink_to(Path(template).resolve())
    return binary, reference


def mean_b0(image, bvals=None, threshold=50):
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("b0 threshold must be finite and nonnegative")
    if image.ndim == 3:
        data = image.get_fdata()
    elif image.ndim == 4:
        if bvals is None:
            raise ValueError("4D DWI requires --bvals")
        values = np.loadtxt(bvals).reshape(-1)
        if len(values) != image.shape[3] or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("b-values must match DWI and be finite/nonnegative")
        indices = np.flatnonzero(values <= threshold)
        if not len(indices):
            raise ValueError("No b0 measurements at the selected threshold")
        # Float64 mean matches the author's get_fdata().mean(-1), bounded memory.
        data = np.zeros(image.shape[:3], dtype=np.float64)
        for index in indices:
            data += np.asarray(image.dataobj[..., int(index)], dtype=np.float64)
        data /= len(indices)
    else:
        raise ValueError("Expected 3D b0 or 4D DWI NIfTI")
    if not np.isfinite(data).all():
        raise ValueError("Input contains NaN/Inf; refusing silent sanitization")
    return np.maximum(data, 0)


def finish_native(template_mask, registered, baseline, matrix, output, *, ants_bin=None, threads=1):
    template_file = Path(output).parent / "template_mask.nii.gz"
    header = registered.header.copy()
    # Upstream uses normalized image header to write the thresholded int16 data.
    nib.save(nib.Nifti1Image(template_mask.astype(np.int16), registered.affine, header), template_file)
    inverse = Path(output).parent / "inverse.nii.gz"
    filled = Path(output).parent / "filled.nii.gz"
    # Default ANTs linear interpolation is intentional. Do not substitute NN.
    run_ants(["antsApplyTransforms", "-d", "3", "-i", template_file, "-r", baseline,
              "-o", inverse, "--transform", f"[{matrix},1]"], ants_bin=ants_bin, threads=threads)
    run_ants(["ImageMath", "3", filled, "FillHoles", inverse], ants_bin=ants_bin, threads=threads)
    original = nib.load(baseline)
    values = nib.load(filled).get_fdata().astype(np.uint8)
    if not np.isin(values, [0, 1]).all() or not values.any():
        raise RuntimeError("Native output must be binary and nonempty")
    header = original.header.copy()
    header.set_data_dtype(np.int16)
    nib.save(nib.Nifti1Image(values, original.affine, header), output)


def run(input_path, output_path, model_dir, template, *, bvals=None, threshold=50, percentile=99,
        device="auto", threads=1, batch_size=1, ants_bin=None, work_dir=None, record=None):
    input_path, output_path = Path(input_path), Path(output_path)
    if output_path.is_symlink() or output_path.resolve() == input_path.resolve():
        raise ValueError("Output must not overwrite the source or follow a symlink")
    if not str(output_path).endswith(".nii.gz"):
        raise ValueError("Output must have the .nii.gz suffix")
    protected = [input_path, Path(template), Path(model_dir) / "weights.json"]
    protected.extend(Path(model_dir) / (view + ".pt") for view in ("axial", "coronal", "sagittal"))
    if bvals:
        protected.append(Path(bvals))
    if any(output_path.resolve() == p.resolve() for p in protected):
        raise ValueError("Output must not overwrite an input or model resource")
    if record and (Path(record).is_symlink() or any(Path(record).resolve() == p.resolve()
                                                  for p in [*protected, output_path])):
        raise ValueError("Record must be separate from input, model resources, and output")
    if threads < 1:
        raise ValueError("Threads must be positive")
    check(template, TEMPLATE_SHA256)
    selection = select_device(device)
    torch.set_num_threads(threads)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.load(input_path, keep_file_open=True)
    baseline_data = mean_b0(image, bvals, threshold)
    with tempfile.TemporaryDirectory(prefix="pnl-torch-", dir=work_dir or ("/tmp" if os.name == "posix" else None)) as temporary:
        work = Path(temporary)
        private_ants, private_template = ants_workspace(work, template, ants_bin)
        baseline = work / "baseline.nii.gz"
        nib.save(nib.Nifti1Image(baseline_data, image.affine, image.header), baseline)
        prefix = work / "rigid-"
        run_ants(["antsRegistrationSyNQuick.sh", "-d", "3", "-f", private_template, "-m", baseline,
                  "-t", "r", "-o", prefix], ants_bin=private_ants, threads=threads)
        registered = nib.load(str(prefix) + "Warped.nii.gz")
        data = normalize(registered.get_fdata(), percentile)
        _, _, mask = predict(data, model_dir, device=selection["resolved"], batch_size=batch_size)
        candidate = work / "native_mask.nii.gz"
        finish_native(mask, registered, baseline, str(prefix) + "0GenericAffine.mat", candidate,
                      ants_bin=private_ants, threads=threads)
        result = nib.load(candidate)
        if result.shape != image.shape[:3] or not np.allclose(result.affine, image.affine, rtol=0, atol=1e-4):
            raise RuntimeError("Native geometry changed")
        with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".nii.gz", delete=False) as stream:
            staged = Path(stream.name)
        try:
            shutil.copyfile(candidate, staged)
            staged.replace(output_path)
        finally:
            staged.unlink(missing_ok=True)
    receipt = {**selection, "package_version": __version__, "source_commit": SOURCE_COMMIT,
               "input_sha256": sha256(input_path), "output_sha256": sha256(output_path),
               "weights": json.loads((Path(model_dir) / "weights.json").read_text()),
               "template_sha256": TEMPLATE_SHA256, "percentile": percentile,
               "view_weights": [0.1, 0.4, 0.5], "threshold": 0.45,
               "inverse_interpolation": "ANTs default linear then FillHoles and uint8 cast",
               "human_anatomical_review": "pending"}
    if record:
        Path(record).write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
