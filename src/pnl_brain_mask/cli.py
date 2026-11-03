"""Standalone inference, author-weight conversion, and asset acquisition."""

import argparse
from importlib.resources import files
import json
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="Download checked converted weights; IIT comes only from NITRC")
    fetch.add_argument("--model-dir", type=Path, required=True)
    fetch.add_argument("--accept-terms", action="store_true")
    fetch.add_argument("--template", action="store_true")
    author = commands.add_parser("fetch-author", help="Acquire original HDF5 weights for offline conversion")
    author.add_argument("--model-dir", type=Path, required=True)
    author.add_argument("--accept-terms", action="store_true")
    converter = commands.add_parser("convert", help="Convert checked HDF5 weights without TensorFlow")
    converter.add_argument("--source", type=Path, required=True)
    converter.add_argument("--output", type=Path, required=True)
    inference = commands.add_parser("mask")
    inference.add_argument("--input", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)
    inference.add_argument("--model-dir", type=Path, required=True)
    inference.add_argument("--template", type=Path)
    inference.add_argument("--bvals", type=Path)
    inference.add_argument("--b0-threshold", type=float, default=50)
    inference.add_argument("--percentile", type=float, default=99)
    inference.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    inference.add_argument("--threads", type=int, default=1)
    inference.add_argument("--batch-size", type=int, default=1)
    inference.add_argument("--ants-bin", type=Path)
    inference.add_argument("--record", type=Path)
    args = parser.parse_args(argv)
    if args.command == "fetch":
        if not args.accept_terms:
            parser.error("Review LICENSE and IIT/NITRC terms and pass --accept-terms")
        from .weights import TEMPLATE_URL, TEMPLATE_SHA256, download
        receipt = json.loads(files("pnl_brain_mask").joinpath("data/converted-weights.json").read_text())
        base = "https://raw.githubusercontent.com/ChengjinLii/PNLBrainMask-Torch/v0.1.0/weights/"
        for record in receipt["views"].values():
            download(base + record["file"], args.model_dir / record["file"], record["sha256"])
        (args.model_dir / "weights.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if args.template:
            download(TEMPLATE_URL, args.model_dir / "IITmean_b0_256.nii.gz", TEMPLATE_SHA256)
    elif args.command == "fetch-author":
        from .weights import acquire_author_weights
        acquire_author_weights(args.model_dir, accept_terms=args.accept_terms)
    elif args.command == "convert":
        from .weights import convert
        print(json.dumps(convert(args.source, args.output), indent=2))
    else:
        if args.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        from .pipeline import run
        run(args.input, args.output, args.model_dir, args.template or args.model_dir / "IITmean_b0_256.nii.gz",
            bvals=args.bvals, threshold=args.b0_threshold, percentile=args.percentile, device=args.device,
            threads=args.threads, batch_size=args.batch_size, ants_bin=args.ants_bin, record=args.record)


if __name__ == "__main__":
    main()
