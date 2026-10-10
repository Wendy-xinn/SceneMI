"""Score native GT against the observed-surface proxy; detect biased geometry."""
import json
from pathlib import Path
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.observed_body_surfaces import from_identity
from experiments.offline_camera_retrain_v1.native_surface_points import NativeSurfacePoints
from experiments.offline_camera_retrain_v1.observed_scene_refinement import geometry_metrics
HERE=Path(__file__).parent;OUT=HERE/'runs/observed_surface_refine_oct10'
def main():
    torch.set_num_threads(4);demo=HERE/'runs/state_relative_spline_oct10/demo';manifest=json.loads((demo/'manifest.json').read_text());rows=[]
    for case in manifest['cases']:
        scene,audit=from_identity(case['identity'],frames=48);skin=NativeSurfacePoints(case['identity']['native_body']);truth=torch.tensor(np.load(demo/f"{case['index']}.npz")['gt_motion'][16:48],device='cuda');metrics=geometry_metrics(truth,skin,scene);rows.append({'index':case['index'],'GT_observed_surface_metrics':metrics,'scope':'future GT mesh only for independent diagnostic scoring, never optimization'});print(case['index'],metrics,flush=True)
    (OUT/'GT_surface_diagnostic.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
