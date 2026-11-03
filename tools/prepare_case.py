"""Prepare a single immutable registered b0 for fair TF/Torch pairing."""

import argparse
from pathlib import Path

import nibabel as nib

from pnl_brain_mask.pipeline import mean_b0, run_ants
from pnl_brain_mask.weights import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--bvals", type=Path)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--ants-bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    image = nib.load(args.input)
    baseline = args.output / "baseline.nii.gz"
    nib.save(nib.Nifti1Image(mean_b0(image, args.bvals), image.affine, image.header), baseline)
    run_ants(["antsRegistrationSyNQuick.sh", "-d", "3", "-f", args.template,
              "-m", baseline, "-t", "r", "-o", args.output / "rigid-"], ants_bin=args.ants_bin, threads=2)
    import json
    receipt = {"input_sha256": sha256(args.input), "bvals_sha256": sha256(args.bvals) if args.bvals else None,
               "baseline_sha256": sha256(baseline), "registered_sha256": sha256(args.output / "rigid-Warped.nii.gz"),
               "matrix_sha256": sha256(args.output / "rigid-0GenericAffine.mat"),
               "template_sha256": sha256(args.template), "threads": 2}
    (args.output / "preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
