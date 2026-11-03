"""Local integration check of the plugin adapter against this installed package."""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin-root", type=Path, required=True)
    parser.add_argument("--dwi", type=Path, required=True)
    parser.add_argument("--bvals", type=Path, required=True)
    parser.add_argument("--bvecs", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--ants-bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.plugin_root))
    from shared.brain_mask import BrainMaskConfig, generate_brain_mask
    config = BrainMaskConfig(backend="pnl_cnn", brain_mask_device="cpu", threads=2,
        pnl_mask_python=Path(sys.executable), pnl_mask_script=Path(sys.executable).parent / "pnl-brain-mask",
        pnl_mask_model_dir=args.models, pnl_mask_ants_bin_dir=args.ants_bin)
    result = generate_brain_mask(args.dwi, args.bvals, args.bvecs, args.output,
                                 config=config, force=True, log=print)
    assert result["device_execution"]["framework"] == "pytorch"
    assert result["device_execution"]["package_version"] == "0.1.0"
    assert result["qc"]["geometry_pass"] and result["qc"]["binary"] and result["qc"]["finite"]
    reused = generate_brain_mask(args.dwi, args.bvals, args.bvecs, args.output, config=config)
    assert reused["status"] == "reused"
    summary = {"status": result["status"], "framework": result["device_execution"]["framework"],
               "package_version": result["device_execution"]["package_version"], "qc": result["qc"],
               "b0_volumes": result["b0_volumes"], "cache_reuse": reused["status"],
               "output_sha256": result["output"]["sha256"], "runtime_seconds": result["runtime_seconds"]}
    args.output.with_name("plugin-smoke.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
