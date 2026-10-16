"""Recover the sealed candidate worker without changing model/training code."""
import argparse
import json
from pathlib import Path
import sys

import io_recovery


ROOT = Path(__file__).resolve().parent


def verify_protocol(path):
    protocol = json.loads(Path(path).read_text())
    for name in ('original_extension_code_sha256', 'recovery_extension_code_sha256'):
        entries = protocol[name]
        if not entries:
            raise ValueError('Missing recovery source hashes: ' + name)
        for filename, expected in entries.items():
            file = Path(filename)
            if not file.is_absolute():
                file = ROOT / file
            if io_recovery.file_sha256(file) != expected:
                raise ValueError('Recovery source changed: ' + str(file))
    return protocol


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--recovery-protocol', required=True)
    own, remaining = parser.parse_known_args()
    verify_protocol(own.recovery_protocol)
    original_argv = sys.argv
    sys.argv = [original_argv[0]] + remaining
    try:
        import worker
        arm_parser = argparse.ArgumentParser(add_help=False)
        arm_parser.add_argument('--arm', required=True)
        args, _ = arm_parser.parse_known_args(remaining)
        effective = json.loads((ROOT/'arms'/args.arm/'prepared/effective_config.json').read_text())
        io_recovery.install(worker, effective, ROOT/'IO_RECOVERY.jsonl')
        worker.main()
        verify_protocol(own.recovery_protocol)
    finally:
        sys.argv = original_argv


if __name__ == '__main__':
    main()
