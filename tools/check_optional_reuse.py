"""Check publicly downloaded optional assets and network-free reuse."""

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plugin-root', type=Path, required=True)
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.plugin_root))
    from scripts.download_model_assets import load_manifest, resolve_asset_plan, verify_asset_plan
    from scripts.setup_optional_assets import prepare
    from shared.optional_assets import MANIFEST
    plan = resolve_asset_plan(load_manifest(MANIFEST), 'pnl-cnn')
    verified = verify_asset_plan(args.asset_root, plan, check_hashes=True)
    assert verified['ok'], verified
    assert all(not name.endswith('.h5') for name in plan.required_paths)
    with patch('scripts.setup_optional_assets.download_asset_plan', side_effect=AssertionError('Unexpected download')):
        reused = prepare(['pnl-cnn'], asset_root=args.asset_root, execute=True, accept_terms=True)
    records = [{'path': name, 'sha256': record['sha256'], 'size_bytes': record['size_bytes'], 'urls': record['urls']}
               for name, record in plan.asset_records.items()]
    summary = {'status': reused['status'], 'profile': 'pnl-cnn', 'hash_verification': 'passed',
               'network_free_reuse': True, 'legacy_hdf5_selected': False, 'files': records}
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
