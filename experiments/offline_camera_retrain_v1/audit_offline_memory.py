"""Recording-union/prefix input equivalence and historical noise provenance.

Read observed arrays and existing records, run contract tests, zero optimizer
steps. This writes a machine audit alongside the single maintained checklist.
"""
import json
import shutil
import time
import unittest

import numpy as np
import torch

from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.native30_data import Native30Data
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder, load_frozen_ego, load_frozen_exo
from experiments.offline_camera_retrain_v1.native30_memory import MEMORY_PROTOCOL, OFFLINE_MEMORY_PROTOCOL
from experiments.offline_camera_retrain_v1.prepare_native30 import read_config, sha256


def main():
    started=time.monotonic();before=shutil.disk_usage(HERE).free
    torch.set_num_threads(2)
    config=read_config();assert config['scene']['memory_protocol']==OFFLINE_MEMORY_PROTOCOL
    suite=unittest.TestSuite(unittest.TestLoader().loadTestsFromName(name) for name in (
        'experiments.offline_camera_retrain_v1.test_native30_protocol_v2',
        'experiments.offline_camera_retrain_v1.test_native30',
        'experiments.offline_camera_retrain_v1.test_exo_occlusion_review'))
    checks=unittest.TextTestRunner(verbosity=1).run(suite)
    if not checks.wasSuccessful():raise RuntimeError('Offline memory regression failed')
    protocol_path=HERE/'runs/ego_exo_surface_continue_oct11/protocol.json'
    legacy=json.loads(protocol_path.read_text());legacy_code={}
    for name in ('unified_observations.py','train_unified_continue.py'):
        actual=sha256(HERE/name)
        if actual!=legacy['code_sha256'][name]:
            raise ValueError('Legacy 3k source no longer matches its recorded training hash')
        legacy_code[name]=actual
    mounts=json.loads((HERE/config['diagnostic_mount_defaults']).read_text())
    skins=Native30SkinAnchorCache(device='cpu');builder=Native30ObservationBuilder(skins,mounts)
    manifest=HERE/config['review_manifest'];real_checks=[]
    for memory_protocol in (OFFLINE_MEMORY_PROTOCOL,MEMORY_PROTOCOL):
        data=Native30Data(manifest,'validation',diagnostic=True,frames=64,memory_protocol=memory_protocol)
        for mode in ('ego','exo'):
            sample,identity=data.sample('camera_wearer',mode,member=0,start=64,yaw=0)
            observed,scene,_=builder.build(data,sample,identity,mode)
            args=dict(start=64,frames=64,skin_cache=skins,device='cpu',memory_manifest=manifest,memory_protocol=memory_protocol)
            frozen=(load_frozen_ego(identity['path'],mounts=mounts,**args) if mode=='ego'
                    else load_frozen_exo(identity['path'],**args))
            np.testing.assert_allclose(scene.points,frozen['surface_windows'][0].points,atol=2e-6)
            np.testing.assert_array_equal(observed['occupancy'],frozen['occupancy'][0].numpy())
            np.testing.assert_allclose(observed['trajectory'][:,15],frozen['trajectory'][0,:,15].numpy(),atol=2e-6)
            real_checks.append(dict(protocol=memory_protocol,mode=mode,sequence_id=identity['sequence_id'],
                source_first=identity['source_first'],window_start=64,window_frames=64,static_points=len(scene.points),
                frozen_matches_training_input=True))
    files=[HERE/p for p in ('native30_memory.py','native30_data.py','native30_observations.py',
        'native30_config.json','prepare_native30.py','train_native30.py','test_native30_protocol_v2.py')]
    files += [manifest,protocol_path,HERE/'runs/native_dynamic_scene20_contact_55k_oct07/config.json',
        HERE/'runs/literature_turn_audit_oct10/summary.json']
    report=dict(date='2026-10-13',optimizer_steps=0,training_ready=False,
        main_memory_protocol=OFFLINE_MEMORY_PROTOCOL,control_memory_protocol=MEMORY_PROTOCOL,
        source_coverage='available exported same-observer segments only; full recording observations not yet exported',
        checks=dict(passed=checks.testsRun,failures=len(checks.failures),errors=len(checks.errors)),
        real_input_equivalence=real_checks,
        legacy_3k_noise=dict(completed_steps=legacy['completed_steps'],code_sha256=legacy_code,
            position_drift_probability=.5,offset_sigma_m=.01,linear_drift_endpoint_sigma_m=.02,
            gap_probability=.3,gap_frames=12,fps=20,rotation_noise=False),
        native30_noise=dict(training_steps=0,offset_sigma_m=.01,linear_drift_endpoint_sigma_m=.02,
            gap_probability=.3,gap_frames=18,fps=30,rotation_noise=False,
            validation='ordinary frozen inputs do not request random noise'),
        input_sha256={str(p):sha256(p) for p in files},
        disk_free_bytes_before=before,disk_free_bytes_after=shutil.disk_usage(HERE).free,
        elapsed_s=time.monotonic()-started)
    path=HERE/'runs/exo_occlusion_review_oct12/OFFLINE_MEMORY_AUDIT.json'
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(checks=report['checks'],real_input_equivalence=real_checks,optimizer_steps=0),ensure_ascii=False))


if __name__=='__main__':main()
