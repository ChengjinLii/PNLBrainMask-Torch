"""Verify cross-environment reuse of public optional PNL assets without network."""

import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plugin-root', type=Path, required=True)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.plugin_root))
    from shared.managed_pnl import provision_assets, verify_assets
    fresh_models = not (args.prefix / 'pnl-models').exists()
    os.environ.update(PIP_NO_INDEX='1', HTTP_PROXY='http://127.0.0.1:9',
                      HTTPS_PROXY='http://127.0.0.1:9', ALL_PROXY='http://127.0.0.1:9', NO_PROXY='')
    with patch('urllib.request.urlopen', side_effect=AssertionError('Unexpected network use')):
        provision_assets(args.prefix, args.cache, asset_root=args.asset_root)
        receipt = verify_assets(args.prefix)
    files = sorted(p.name for p in (args.prefix / 'pnl-models').iterdir())
    assert not any(name.endswith('.h5') for name in files)
    summary = {'status': 'passed', 'fresh_model_directory': fresh_models, 'network_disabled': True,
               'files': files, 'legacy_hdf5_files': [], 'compatible_torch_preserved': True,
               'torch_version': receipt['pnl_runtime_probe']['torch_version'],
               'forward_checks': receipt['pnl_runtime_probe']['forward_checks']}
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
