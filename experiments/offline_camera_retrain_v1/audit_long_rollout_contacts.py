"""RICH generated-motion regional contact scoring, not physical ground truth.

Same GT reference labels score reconstruction-associated contacts. They do not
certify that a different generated motion physically contacts those surfaces.
"""
import json
import numpy as np
import torch
from torch import nn
from experiments.offline_camera_retrain_v1.evaluate_long_history_rollout import HERE,OUT
from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
from experiments.offline_camera_retrain_v1.contact_supervision import vertex_regions,aggregate_contacts

def main():
    torch.set_num_threads(2);protocol=json.loads((OUT/'protocol.json').read_text());cp=torch.load(protocol['weights']['path'],map_location='cpu',weights_only=False)
    head=nn.Sequential(nn.Linear(402,128),nn.SiLU(),nn.Linear(128,22));head.load_state_dict({k[len('contact_head.'):]:v for k,v in cp['model'].items() if k.startswith('contact_head.')});root=cp['config']['rich_contact_root'];del cp;head.eval();rows=[]
    for i,identity in enumerate(protocol['identities']):
        if identity['group']!='rich':continue
        from pathlib import Path
        folder=Path(identity['scene_bundle']);ids=np.load(folder/'source_frame_ids.npy');start=int(np.searchsorted(ids,identity['source_start_30fps']));body=identity['native_body']
        with np.load(Path(root)/'val'/identity['sequence_id']/'contacts.npz') as labels:
            assert np.allclose(labels['source_frame_ids'][start:start+176],ids[start:start+176])
            target,valid=aggregate_contacts(labels['smplx_packed'],labels['valid'][:,1],vertex_regions(load_model(body['model'],body['gender'])),10475)
        target=torch.tensor(target[start+16:start+176]).bool();mask=torch.tensor(valid[start+16:start+176]);bps=np.load(folder/'causal_bps_20.npy')[start+16:start+176];bv=np.load(folder/'causal_bps_valid_20.npy')[start+16:start+176];features=torch.tensor((bps*bv[...,None]).reshape(160,201))
        for rep in range(2):
            sources=[OUT/f'motions_rep{rep}.npz',OUT/f'warm_motions_rep{rep}.npz']
            for source in sources:
                with np.load(source) as arrays:
                    for label in arrays.files:
                        if label=='gt':continue
                        motion=torch.tensor(arrays[label][i,16:176])
                        with torch.no_grad():pred=head(torch.cat((motion,features),-1)).sigmoid()>=.5
                        tp=(pred&target&mask).sum(0).numpy();fp=(pred&~target&mask).sum(0).numpy();fn=(~pred&target&mask).sum(0).numpy()
                        rows.append({'sequence_id':identity['sequence_id'],'replicate':rep,'label':label,'labeled_entries':int(mask.sum()),'TP_per_region':tp.tolist(),'FP_per_region':fp.tolist(),'FN_per_region':fn.tolist(),'micro_f1':float(2*tp.sum()/max(2*tp.sum()+fp.sum()+fn.sum(),1)),'support_per_region':(target&mask).sum(0).tolist()})
    (OUT/'contact_audit.json').write_text(json.dumps({'scope':'2 identity/coverage-selected RICH recordings, generated motions, 8s, two seeds; reconstruction-associated reference labels, not scene collision certification','contact_target_input':False,'rows':rows},indent=2));print('Regional contact audit rows',len(rows))
if __name__=='__main__':main()
