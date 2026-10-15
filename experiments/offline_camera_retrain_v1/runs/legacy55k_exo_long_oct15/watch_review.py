"""Rebuild only when a new validation prediction is atomically published."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    last=None
    while True:
        path=ROOT/'validation_latest.pt'
        stamp=path.stat().st_mtime_ns if path.exists() else None
        if stamp is not None and stamp!=last:
            p=subprocess.run([sys.executable,str(ROOT/'review_live.py')])
            if p.returncode==0:last=stamp
        status=ROOT/'STATUS.json'
        if status.exists() and json.loads(status.read_text())['status'] in ('completed','stopped_at_update_boundary','failed_saved'):break
        time.sleep(20)
