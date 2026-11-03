"""Exercise real managed installation and reuse without editing user settings."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--conda", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.plugin_root))
    from shared.managed_runtimes import PROFILES, provision
    from shared.managed_pnl import ASSETS, PNL_TORCH_COMMIT, PNL_WHEEL, pnl_python_environment

    prefix = args.runtime_root / PROFILES["pnl_cnn"]["directory"]
    fresh = not prefix.exists()
    started = time.monotonic()
    first = provision(("pnl_cnn",), root=args.runtime_root, conda=args.conda,
                      accept_pnl_terms=True, asset_root=args.asset_root)
    installed_seconds = time.monotonic() - started
    second = provision(("pnl_cnn",), root=args.runtime_root, conda=args.conda,
                       accept_pnl_terms=True, asset_root=args.asset_root)
    assert first["external_dependencies"] == second["external_dependencies"]
    receipt = json.loads(Path(first["receipts"]["pnl_cnn"]).read_text())
    assert receipt["status"] == "ready" and receipt["pnl_framework"] == "pytorch"
    assert receipt["pnl_torch_commit"] == PNL_TORCH_COMMIT
    query = """import importlib.util,json,numpy,torch,pnl_brain_mask
absent={p:importlib.util.find_spec(p) is None for p in ('tensorflow','keras','h5py','conversion')}
assert all(absent.values()),absent
print(json.dumps({'absent_offline_dependencies':absent,'torch_version':torch.__version__,
'numpy_version':numpy.__version__,'package_version':pnl_brain_mask.__version__}))
"""
    check = subprocess.run([str(prefix / "bin/python"), "-c", query], check=True,
                           env=pnl_python_environment(prefix / "bin/python"), capture_output=True, text=True)
    summary = {"fresh_install": fresh, "status": first["status"], "receipt_status": receipt["status"],
               "reuse_same_configuration": True, "pnl_torch_commit": PNL_TORCH_COMMIT,
               "wheel_url": ASSETS[PNL_WHEEL][0], "wheel_sha256": ASSETS[PNL_WHEEL][1],
               "functional_checks": receipt["functional_checks"],
               "forward_checks": receipt["pnl_runtime_probe"]["forward_checks"],
               "runtime": json.loads(check.stdout), "installation_seconds": installed_seconds}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
