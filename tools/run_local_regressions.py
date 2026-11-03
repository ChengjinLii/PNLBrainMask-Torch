"""Run PNL-facing plugin regressions locally; no CI service is used."""

import argparse
import json
from pathlib import Path
import sys
import unittest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plugin-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tests = args.plugin_root / 'Testing/Python'
    sys.path[:0] = [str(args.plugin_root), str(tests)]
    patterns = ['test_brain_mask*.py', 'test_managed_pnl.py', 'test_gpu_execution_policy.py',
                'test_preprocessing_mask_integration.py', 'test_managed_runtimes.py',
                'test_environment*.py', 'test_extension_distribution.py', 'test_optional_assets.py']
    suite = unittest.TestSuite()
    for pattern in patterns:
        suite.addTests(unittest.defaultTestLoader.discover(str(tests), pattern=pattern))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    summary = {'patterns': patterns, 'python': sys.version.split()[0], 'tests_run': result.testsRun,
               'failures': len(result.failures), 'errors': len(result.errors),
               'skipped': [{'test': str(test), 'reason': reason} for test, reason in result.skipped],
               'passed': result.wasSuccessful()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
