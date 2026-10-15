"""Read-only progress entry point; reports stale processes and saved step."""
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent

if __name__=='__main__':
    path=ROOT/'STATUS.json'
    if path.exists():
        status=json.loads(path.read_text())
        try:os.kill(status['pid'],0);status['process_alive']=True
        except ProcessLookupError:status['process_alive']=False
        except PermissionError:status['process_alive']=None
        print(json.dumps(status,indent=2,ensure_ascii=False))
    else:
        protocol=ROOT/'PROTOCOL.json'
        print(json.dumps(json.loads(protocol.read_text()) if protocol.exists() else {'status':'not_started'},indent=2,ensure_ascii=False))
