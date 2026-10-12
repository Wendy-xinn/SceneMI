"""Build official raw-source index; record bounded split/memory checks, no training."""
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
import unittest

from experiments.offline_camera_retrain_v1.data import HERE, SOURCE
from experiments.offline_camera_retrain_v1.native30_export import inventory, DEFAULT_OUTPUT
from experiments.offline_camera_retrain_v1.native30_splits import SPLITS, SPLIT_PROTOCOL, OFFICIAL_CSV, split_summary
from experiments.offline_camera_retrain_v1.native30_memory import MEMORY_PROTOCOL


def main():
    before=shutil.disk_usage(HERE).free
    sources=[r for split in SPLITS for r in inventory(split)]
    DEFAULT_OUTPUT.mkdir(parents=True,exist_ok=True)
    index=DEFAULT_OUTPUT/'source_index.jsonl'
    temp=index.with_suffix('.tmp')
    temp.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in sources))
    temp.replace(index)
    suite=unittest.TestLoader().loadTestsFromName('experiments.offline_camera_retrain_v1.test_native30_protocol_v2')
    checks=unittest.TextTestRunner(verbosity=1).run(suite)
    if not checks.wasSuccessful():raise RuntimeError('Official split / memory audit failed')
    dependencies=[HERE/p for p in ('native30_splits.py','native30_memory.py','native30_data.py',
        'native30_export.py','native30_observations.py','native30_geometry.py','native30_config.json',
        'test_native30_protocol_v2.py')]+[OFFICIAL_CSV,index,SOURCE/'sequences.jsonl']+[
            SOURCE/f'egobody_sequences/{split}.jsonl' for split in SPLITS]
    report=dict(split_protocol=SPLIT_PROTOCOL,memory_protocol=MEMORY_PROTOCOL,optimizer_steps=0,
        official_egobody=split_summary(),source_index=str(index),
        source_counts={split:dict(Counter(r['group'] for r in sources if r['split']==split)) for split in SPLITS},
        checks=dict(passed=checks.testsRun,errors=len(checks.errors),failures=len(checks.failures)),
        memory_budget='same-observer observed static prefix through window end; dynamics current frame',
        memory_scope='available exported observation segments only; no surfaces from gaps/full scans',
        training_ready=False,pending=['full_raw30_export','official_TRAIN_camera_mount_refit',
            'GT_quality_masks','observation_noise_and_loss_config','formal_validation','visual_acceptance'],
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies},
        disk_free_bytes_before=before,disk_free_bytes_after=shutil.disk_usage(HERE).free)
    path=HERE/'runs/exo_occlusion_review_oct12/PROTOCOL_V2_AUDIT.json'
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:report[k] for k in ('official_egobody','source_counts','checks','training_ready')},ensure_ascii=False))


if __name__=='__main__':main()
