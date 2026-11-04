"""Resume a pinned EXO export with one fresh geometry process per recording.

Native geometry faults cannot leave the coordinator claiming to be running.
Only hash-verified, semantically checked atomic metadata commits are accepted.
The existing exporter, observation policy and completed data remain unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from scenemi.data.exo_export import atomic_json, array_path, candidates, export_one, verify_output
from scenemi.native_visualization import ROOT, sha256


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_record(row, folder):
    """Validate both file identities and the public observation semantics."""
    meta = json.loads((folder/'metadata.json').read_text())
    if meta.get('diagnostic_only'):
        raise ValueError('A diagnostic pilot cannot be resumed as full training data')
    for key in ('sequence_id','split','frames'):
        if meta[key] != row[key]:
            raise ValueError(f'Recovered record identity mismatch: {key}')
    if meta['source_native_path'] != row['path']:
        raise ValueError('Recovered record points at another native source')
    if meta['source_native_metadata_sha256'] != sha256(Path(row['path'])/'metadata.json'):
        raise ValueError('Original native source metadata changed')
    verify_output(meta)
    def load(name):
        return np.load(array_path(meta,name+'.npy'),mmap_mode='r',allow_pickle=False)
    n = row['frames'];available = load('exo_position_available');confidence = load('exo_position_confidence')
    position = load('exo_trajectory_world');valid = load('observation_valid')
    if available.shape != (n,22) or confidence.shape != (n,22) or position.shape != (n,22,3):
        raise ValueError('Recovered joint stream has the wrong clock')
    reference,visible = load('exo_reference_counts'),load('exo_visible_counts')
    blocked = sum(load('exo_'+k+'_counts') for k in ('person','static','rigid'))
    if not np.array_equal(reference,visible+blocked):
        raise ValueError('Recovered occlusion partition is incomplete')
    fraction = visible/np.maximum(reference,1)
    expected = (visible>=2)&(fraction>=.15)&valid[:,None]
    expected_confidence = np.where(expected,fraction*np.minimum(visible/12.,1),0)
    if not np.array_equal(available,expected) or not np.allclose(confidence,expected_confidence,atol=1e-6):
        raise ValueError('Recovered availability/confidence disagrees with the ray evidence')
    if not np.isfinite(position).all() or np.any(position[~available]) or np.any(confidence[~available]):
        raise ValueError('Recovered unavailable joints expose positions or confidence')
    target = load('motion_world')[:,135:].reshape(n,22,3)*2
    if not np.allclose(position[available],target[available],atol=1e-5):
        raise ValueError('Recovered visible joint positions use another world frame')
    owners,points = load('exo_frame_owner'),load('exo_frame_points_world')
    if owners.shape != (n,512) or points.shape != (n,512,3) or np.any(owners==100):
        raise ValueError('Recovered environment leaks target skin or has the wrong clock')
    if not np.isfinite(points).all() or np.any(points[owners<0]):
        raise ValueError('Recovered environment has invalid point payloads')
    return meta


def worker(output, index, progress_file):
    import torch
    from experiments.offline_camera_retrain_v1.data import PROJECTS
    from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
    from experiments.offline_camera_retrain_v1.export_egobody_exo import regional_faces
    from scenemi.training import ensure_bounded

    ensure_bounded();torch.set_num_threads(1)
    contract = json.loads((output/'PROTOCOL.json').read_text())
    manifest = Path(json.loads((output/'PLAN.json').read_text())['source_manifest'])
    row = candidates(manifest)[index]
    model = load_model('smplx','male')
    labels = regional_faces(model)
    clips = {r['clip_name']:r for r in map(json.loads,
        (PROJECTS/'TRUMANS/processed/scene_expert_v1/clips.jsonl').read_text().splitlines())}
    def progress(done,total):
        atomic_json(progress_file,dict(pid=os.getpid(),sequence_id=row['sequence_id'],
            current_frames=done,current_sequence_frames=total,updated_at_utc=now()))
    folder = output/'data'/row['split']/row['sequence_id']
    export_one(row,folder,contract['policy'],model,labels,clips,progress,diagnostic=False)
    print(json.dumps(dict(state='committed',sequence_id=row['sequence_id'],frames=row['frames'])),flush=True)


def resume(output):
    from scenemi.training import ensure_bounded
    ensure_bounded()
    contract_path = output/'PROTOCOL.json'
    contract = json.loads(contract_path.read_text())
    plan = json.loads((output/'PLAN.json').read_text())
    manifest = Path(plan['source_manifest'])
    if contract['limit'] is not None or sha256(manifest) != contract['source_manifest_sha256']:
        raise ValueError('Resume requires the original complete train/validation export')
    for name,digest in contract['sources'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError('Sealed exporter dependency changed: '+name)
    if sha256(ROOT/'body_models/smplx/SMPLX_MALE.npz') != contract['native_body_sha256']:
        raise ValueError('Native body asset changed')
    recovery = output/'recovery';recovery.mkdir(exist_ok=True)
    recovery_contract = dict(protocol='exo-recording-isolated-recovery-v1',
        parent_protocol_sha256=sha256(contract_path),worker_source_sha256=sha256(__file__),
        original_observation_policy_unchanged=True,max_attempts_per_recording=2,
        geometry_process_lifetime='one complete source recording',test_used=False,training_started=False)
    recovery_contract_path = recovery/'PROTOCOL.json'
    if recovery_contract_path.exists() and json.loads(recovery_contract_path.read_text()) != recovery_contract:
        raise ValueError('Recovery source/protocol changed')
    atomic_json(recovery_contract_path,recovery_contract)
    rows = candidates(manifest)
    state = json.loads((output/'STATUS.json').read_text())
    state.update(state='verifying_completed',coordinator_pid=os.getpid(),worker_pid=None,
        resumed_at_utc=now(),updated_at_utc=now(),verified_sequences=0,
        current_frames=0,current_sequence_frames=0,rendered_frames_this_invocation=0,
        worker_protocol=recovery_contract['protocol'],eta_seconds=None)
    atomic_json(output/'STATUS.json',state)
    committed,missing = [],[]
    current_child = None
    failures = []
    def save_manifest():
        temporary = output/'manifest.jsonl.tmp'
        with temporary.open('w') as stream:
            for item in sorted(committed,key=lambda r:(r['split'],r['sequence_id'])):
                stream.write(json.dumps(item,ensure_ascii=False)+'\n')
        temporary.replace(output/'manifest.jsonl')
    def stop(signum,_):
        if current_child is not None and current_child.poll() is None:
            current_child.terminate()
        raise KeyboardInterrupt(f'Recovery coordinator received signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    try:
        for index,row in enumerate(rows):
            folder = output/'data'/row['split']/row['sequence_id']
            if (folder/'metadata.json').exists():
                committed.append(verify_record(row,folder))
                state.update(verified_sequences=len(committed),updated_at_utc=now())
                atomic_json(output/'STATUS.json',state)
            else:
                missing.append((index,row))
        save_manifest()
        verified_frames = sum(r['frames'] for r in committed)
        state.update(state='preparing',completed_sequences=len(committed),completed_frames=verified_frames,
            reused_frames=verified_frames,updated_at_utc=now())
        atomic_json(output/'STATUS.json',state)
        started = time.monotonic();fresh_frames = 0
        for index,row in missing:
            folder = output/'data'/row['split']/row['sequence_id']
            for attempt in range(1,3):
                progress_file = recovery/'WORKER_PROGRESS.json'
                atomic_json(progress_file,dict(current_frames=0,current_sequence_frames=row['frames']))
                log_path = recovery/(row['split']+'_'+row['sequence_id']+f'_attempt{attempt}.log')
                command = [sys.executable,'-m','scenemi.data.exo_workers','worker','--output',str(output),
                           '--index',str(index),'--progress-file',str(progress_file)]
                with log_path.open('a') as log:
                    current_child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                    last_done = -1;last_progress = time.monotonic()
                    while current_child.poll() is None:
                        progress = json.loads(progress_file.read_text())
                        done = progress['current_frames'];elapsed = time.monotonic()-started
                        if done != last_done:
                            last_done = done;last_progress = time.monotonic()
                        elif time.monotonic()-last_progress > 180:
                            log.write('\nCoordinator: worker made no progress for 180 seconds.\n');log.flush()
                            current_child.terminate()
                            try:
                                current_child.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                current_child.kill();current_child.wait(timeout=15)
                            break
                        speed = (fresh_frames+done)/max(elapsed,1e-6)
                        remaining = max(0,state['total_frames']-state['completed_frames']-done)
                        state.update(state='preparing',sequence_id=row['sequence_id'],split=row['split'],
                            current_frames=done,current_sequence_frames=row['frames'],worker_pid=current_child.pid,
                            worker_attempt=attempt,updated_at_utc=now(),frames_per_second=speed,
                            eta_seconds=remaining/speed if speed else None,
                            rendered_frames_this_invocation=fresh_frames+done)
                        atomic_json(output/'STATUS.json',state)
                        time.sleep(3)
                    return_code = current_child.returncode
                current_child = None
                if return_code != 0:
                    incident = dict(sequence_id=row['sequence_id'],split=row['split'],attempt=attempt,
                        return_code=return_code,log=str(log_path),updated_at_utc=now(),
                        metadata_committed=(folder/'metadata.json').exists())
                    failures.append(incident);atomic_json(recovery/'WORKER_FAILURES.json',failures)
                if (folder/'metadata.json').exists():
                    committed.append(verify_record(row,folder));break
                if attempt==2:
                    raise RuntimeError(f'Two workers failed for {row["sequence_id"]}; no data were skipped')
            fresh_frames += row['frames']
            state.update(completed_sequences=len(committed),completed_frames=sum(r['frames'] for r in committed),
                current_frames=row['frames'],rendered_frames_this_invocation=fresh_frames,worker_pid=None,
                worker_incidents=len(failures),updated_at_utc=now())
            save_manifest();atomic_json(output/'STATUS.json',state)
            print(json.dumps({k:state[k] for k in ('completed_sequences','sequences','completed_frames','total_frames','eta_seconds')}),flush=True)
        for name,digest in contract['sources'].items():
            if sha256(ROOT/name) != digest:
                raise ValueError('Exporter source changed during recovery')
        if sha256(__file__) != recovery_contract['worker_source_sha256']:
            raise ValueError('Recovery coordinator source changed')
        state.update(state='complete',full_dataset_reexported=True,eta_seconds=0.,worker_pid=None,updated_at_utc=now())
        atomic_json(output/'STATUS.json',state)
    except BaseException as error:
        if current_child is not None and current_child.poll() is None:
            current_child.terminate();current_child.wait(timeout=15)
        state.update(state='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
            error_type=type(error).__name__,error=str(error),updated_at_utc=now(),worker_pid=None)
        atomic_json(output/'STATUS.json',state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('resume','worker'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--index',type=int)
    parser.add_argument('--progress-file',type=Path)
    args = parser.parse_args()
    if args.action=='worker':
        if args.index is None or args.progress_file is None:
            parser.error('worker requires index and progress-file')
        worker(args.output,args.index,args.progress_file)
    else:
        resume(args.output)


if __name__=='__main__':
    main()
