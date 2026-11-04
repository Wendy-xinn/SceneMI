"""Replay-stage preparation and measured throughput; no implicit training launch."""
import os
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def main():
    if len(sys.argv)>1 and sys.argv[1] in ('run','check'):
        from .runtime import main as action
    elif len(sys.argv)>1 and sys.argv[1]=='metrics-benchmark':
        from .metrics import main as action
    elif len(sys.argv)>1 and sys.argv[1]=='pilot':
        from .pilot import main as action
    else:
        from .benchmark import main as action
    action()


if __name__=='__main__':main()
