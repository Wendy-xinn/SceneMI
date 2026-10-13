"""Resumable bounded-process export of all indexed raw30 sources to external storage."""
import argparse
import contextlib
import ctypes
import faulthandler
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from experiments.offline_camera_retrain_v1.native30_cache_io import atomic_text,mapped_sha256,read_json_retry,read_text_retry

VERIFY_LOCK=threading.Lock()


def write_json(path,value):
    atomic_text(path,json.dumps(value,indent=2))


def worker(row,output):
    from experiments.offline_camera_retrain_v1.native30_export import export
    log=Path(output)/'logs'/row['split']/(row['sequence_id']+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',buffering=1) as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        faulthandler.enable(file=stream,all_threads=True)
        scratch=Path('/tmp/scenemi_native30_export_staging')/str(os.getpid())
        final=Path(output)/row['split']/row['sequence_id']
        try:
            if (final/'metadata.json').exists():
                try:return export(row,Path(output))
                except (ValueError,FileNotFoundError) as exc:
                    if isinstance(exc,ValueError) and not isinstance(exc,json.JSONDecodeError) and not str(exc).startswith('Corrupt native30 export:'):
                        raise  # Never silently substitute a different export protocol.
                    # A host restart can interrupt the final filesystem flush.
                    # Preserve the damaged generated bundle for diagnosis.
                    quarantine=Path(output)/'_interrupted_bundles'/row['split']/(row['sequence_id']+f'_{time.time_ns()}')
                    quarantine.parent.mkdir(parents=True,exist_ok=True);final.replace(quarantine)
                    print('Preserved interrupted cache for reconstruction',str(quarantine),repr(exc),flush=True)
            # DrvFS/9p msync can return ENOMEM despite Linux available memory.
            # Keep only an active bundle on ext4, then copy/commit it to E.
            for attempt in range(3):
                try:
                    result=export(row,scratch);break
                except ValueError as exc:
                    if 'invalid \\x escape' not in str(exc) or attempt==2:raise
                    print('Retrying transient protocol-0 model read',attempt+1,repr(exc),flush=True)
                    time.sleep(1)
            generated=Path(result['path']);temporary=final.with_name(final.name+'.copying')
            if temporary.exists():shutil.rmtree(temporary)
            shutil.copytree(generated,temporary)
            if final.exists():shutil.rmtree(final)  # this run's incomplete bundle only
            temporary.replace(final)
            result['path']=str(final);return result
        except Exception:
            traceback.print_exc();raise
        finally:
            if scratch.exists():shutil.rmtree(scratch)
            gc.collect()
            ctypes.CDLL('libc.so.6').malloc_trim(0)
            faulthandler.disable()


def verify_existing(row,output):
    """Streaming verification needs no reusable Open3D process or model."""
    with VERIFY_LOCK:return _verify_existing(row,output)


def _verify_existing(row,output):
    from experiments.offline_camera_retrain_v1.native30_export import PROTOCOL,export_fingerprint
    final=Path(output)/row['split']/row['sequence_id']
    if not (final/'metadata.json').exists():return None
    try:
        meta=read_json_retry(final/'metadata.json')
        if meta['protocol']!=PROTOCOL or meta.get('export_fingerprint')!=export_fingerprint():
            raise ValueError('Stale native30 export; use a new directory')
        for key,digest in meta['output_sha256'].items():
            if mapped_sha256(final/key)!=digest:raise ValueError(f'Corrupt native30 export: {key}')
        return dict(path=str(final),**meta)
    except (ValueError,FileNotFoundError) as exc:
        if isinstance(exc,ValueError) and not isinstance(exc,json.JSONDecodeError) and not str(exc).startswith('Corrupt native30 export:'):
            raise
        quarantine=Path(output)/'_interrupted_bundles'/row['split']/(row['sequence_id']+f'_{time.time_ns()}')
        quarantine.parent.mkdir(parents=True,exist_ok=True);final.replace(quarantine)
        print('Preserved interrupted cache for reconstruction',str(quarantine),repr(exc),flush=True)
        return None


def isolated_worker(row,output):
    # A native-library crash must affect only its source. Python3.9's executor
    # cannot retire workers after one task, so launch one fresh subprocess per
    # new bundle. Existing bundles only need streaming hashes in these threads.
    existing=verify_existing(row,output)
    if existing is not None:return existing
    with tempfile.TemporaryDirectory(prefix='scenemi_native30_worker_') as temporary:
        temporary=Path(temporary);request=temporary/'row.json';result=temporary/'result.json'
        request.write_text(json.dumps(row))
        command=[sys.executable,'-u','-m','experiments.offline_camera_retrain_v1.export_native30_all',
                 '--output',str(output),'--worker-row',str(request),'--worker-result',str(result)]
        log=Path(output)/'logs'/row['split']/(row['sequence_id']+'.launcher.log')
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as stream:
            for attempt in range(3):
                process=subprocess.Popen(command,stdout=stream,stderr=stream)
                returncode=process.wait()
                # A signal can bypass worker.finally; only this child's scratch
                # is expendable. A copied final bundle remains hash-verified.
                scratch=Path('/tmp/scenemi_native30_export_staging')/str(process.pid)
                if scratch.exists():shutil.rmtree(scratch)
                if returncode==0:return json.loads(result.read_text())
                if returncode not in (-11,-6) or attempt==2:
                    raise RuntimeError(f'Isolated source export exited {returncode}: {row["sequence_id"]}; see {log}')
                print('Retrying isolated native-library crash',row['sequence_id'],returncode,attempt+1,flush=True)


def initialize_launcher(cpus,workers,slots):
    if hasattr(os,'sched_setaffinity'):
        slot=next(slots);per_worker=max(1,len(cpus)//workers)
        # Linux affinity applies to the launcher thread; its fresh child
        # inherits it, keeping Open3D's internal parallelism bounded.
        os.sched_setaffinity(0,[cpus[(per_worker*slot+i)%len(cpus)] for i in range(per_worker)])


def main():
    from experiments.offline_camera_retrain_v1.native30_export import inventory,export_fingerprint
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--worker-row',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--worker-result',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if args.worker_row:
        write_json(args.worker_result,worker(json.loads(args.worker_row.read_text()),output));return
    if args.workers<1:parser.error('--workers must be positive')
    rows=[r for s in ('train','validation','test') for r in inventory(s)]
    source_index=output/'source_index.jsonl'
    if source_index.exists():
        if [json.loads(line) for line in source_index.read_text().splitlines()]!=rows:
            raise ValueError('Existing source index differs from full native30 inventory')
    else:
        temporary=source_index.with_suffix('.tmp')
        temporary.write_text(''.join(json.dumps(row)+'\n' for row in rows));temporary.replace(source_index)
    # Existing complete bundles are verified by export(), then incorporated.
    # Incomplete bundles have no metadata commit marker and are rebuilt safely.
    expected={(r['split'],r['sequence_id']) for r in rows}
    manifest=output/'manifest.jsonl';existing={}
    if manifest.exists():
        existing={(r['split'],r['bundle_id']):r for r in map(json.loads,read_text_retry(manifest).splitlines())}
        if set(existing)-expected:raise ValueError('Manifest includes stale/unexpected sources')
    estimated_bytes=sum(r['raw_last_exclusive']-r['raw_first'] for r in rows)*18500
    free=shutil.disk_usage(output).free
    if free<estimated_bytes+20*2**30:raise ValueError('Insufficient destination space with 20 GiB reserve')
    run=dict(pid=os.getpid(),started_unix=time.time(),output=str(output),workers=args.workers,
        optimizer_steps=0,expected_sources=len(rows),expected_frames=sum(r['raw_last_exclusive']-r['raw_first'] for r in rows),
        conservative_output_bytes=estimated_bytes,free_before=free,export_fingerprint=export_fingerprint())
    write_json(output/'export_run.json',run);errors=[];done=0;frames=0;started=time.monotonic()
    def progress(status):
        report=dict(run,status=status,completed_sources=done,completed_frames=frames,errors=errors,
            elapsed_s=time.monotonic()-started,free_now=shutil.disk_usage(output).free)
        write_json(output/'export_progress.json',report)
    def save_manifest():
        atomic_text(manifest,''.join(json.dumps(existing[k])+'\n' for k in sorted(existing)))
    progress('running')
    # Short gaps first commit complete-recording observer coverage cheaply.
    rows.sort(key=lambda r:(r['raw_last_exclusive']-r['raw_first'],r['split'],r['sequence_id']))
    cpus=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else list(range(os.cpu_count()))
    with ThreadPoolExecutor(max_workers=args.workers,initializer=initialize_launcher,
            initargs=(cpus,args.workers,iter(range(args.workers)))) as pool:
        pending={};todo=iter(rows)
        def replenish():
            while len(pending)<args.workers*2:
                row=next(todo,None)
                if row is None:break
                pending[pool.submit(isolated_worker,row,str(output))]=row
        replenish()
        while pending:
            finished,_=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
            for future in finished:
                row=pending.pop(future)
                try:
                    result=future.result();existing[(result['split'],result['bundle_id'])]=result
                    done+=1;frames+=result['frames'];save_manifest()
                    print(f"{done}/{len(rows)} sources, {frames}/{run['expected_frames']} frames: {row['split']}/{row['sequence_id']}",flush=True)
                except Exception as exc:
                    errors.append(dict(split=row['split'],sequence_id=row['sequence_id'],error=repr(exc)))
                    print('EXPORT ERROR',errors[-1],flush=True)
                    traceback.print_exc()
            replenish();progress('running')
    for split in ('train','validation','test'):
        scoped=[r for r in rows if r['split']==split]
        covered=[existing[(split,r['sequence_id'])] for r in scoped if (split,r['sequence_id']) in existing]
        write_json(output/f'{split}_coverage.json',dict(split=split,complete=len(covered)==len(scoped),
            expected=len(scoped),exported=len(covered),frames=sum(r['frames'] for r in covered),
            source_time_coverage='full master recording; ego PV segments separate from exo clock',
            modes={mode:sum(mode in r['allowed_modes'] for r in covered) for mode in ('ego','exo')}))
    progress('complete' if not errors and done==len(rows) else 'failed')
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
