"""Historical identity utilities; GPU launch admission no longer requires a registry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from scenemi import parallel_resources
from scenemi.config import ROOT


def identity(pid):
    proc=Path('/proc')/str(pid)
    stat=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,uid=proc.stat().st_uid,start_ticks=int(stat[19]),
        cwd=str(proc.joinpath('cwd').resolve()),executable=str(proc.joinpath('exe').resolve()),
        command_sha256=hashlib.sha256(proc.joinpath('cmdline').read_bytes()).hexdigest())


def authorized_job(pid, records):
    try:
        actual=identity(pid)
        return actual['uid']==os.getuid() and any(actual==record for record in records)
    except (OSError,ValueError,IndexError):return False


def install(registry):
    from scenemi.gpu_policy import load_policy
    load_policy()
    parallel_resources.MODULES |= {'scenemi.scene_followup','scenemi.scene_checks','scenemi.scene_relations','scenemi.continuation','scenemi.native_ego4_training'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry',type=Path,default=ROOT/'configs/gpu_shared_jobs.local.json')
    args,remaining=p.parse_known_args()
    install(args.registry)
    if remaining and remaining[0]=='watch':
        from scenemi.training import ensure_bounded
        from scenemi.scene_checks import watch
        ensure_bounded();q=argparse.ArgumentParser()
        q.add_argument('--policy',type=Path,default=ROOT/'configs/scene_checkpoint_3k.json')
        opt=q.parse_args(remaining[1:]);return watch(opt.policy)
    sys.argv=[__file__]+remaining
    parallel_resources.main()


if __name__=='__main__':main()
