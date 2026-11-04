"""Small real-data acceptance for source-only GRAIL contact weights.

Writes compact training-label sidecars and evidence. It does not start a scene
run, regenerate the corpus, alter selected ranges, or open test arrays.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from scenemi.config import ROOT
from scenemi.contact_loss import DEFAULT_POLICY, load_policy, contact_reliability, contact_terms
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.checked_scene_dataset import SceneSelection
from scenemi.data.grail_contact import ContactBody, load_source
from scenemi.data.grail_quality import contact_probabilities
from scenemi.data.grail_support import sole_measure
from scenemi.training import ensure_bounded, now
from experiments.offline_camera_retrain_v1.ops.grail_source import load_pickle
from experiments.offline_camera_retrain_v1.ops.viewer_geometry import readonly_array


def training_population(bundle):
    rows = [r for name in bundle.policy['grail_selection']['ego']['train'] for r in bundle.rows(name)]
    families = defaultdict(list)
    for row in rows:
        families[row['object_family_id']].append(row['category'])
    probability = Counter()
    for categories in families.values():
        for category in categories:
            probability[category] += 1./len(categories)/len(families)
    counts = Counter(r['category'] for r in rows)
    return rows, {category:dict(sequences=counts[category],
        probability_within_grail=p,expected_samples_in_32_grail_quota=32*p)
        for category,p in probability.items()}


def audit(args):
    ensure_bounded();torch.set_num_threads(2)
    policy = load_policy(args.config)
    bundle = SceneSelection(ROOT/'configs/scene_data_current.json')
    rows,population = training_population(bundle)
    # Fixed selection by source category/path before measuring reliability.
    choices = [min((r for r in rows if r['category']==c),key=lambda r:r['path'])
               for c in ('curb','slope','stair_p1','stair_p2')]
    support = json.loads((ROOT/'configs/grail_ego_support_v2.json').read_text())
    source_policy = json.loads((ROOT/'configs/grail_ego_contact.json').read_text())
    body = ContactBody(source_policy)
    evidence = []
    args.output.mkdir(parents=True,exist_ok=True)
    for row in choices:
        meta,params,rotation,offset = load_source(row)
        if meta['split'] != 'train':raise ValueError('Acceptance is train-only')
        for key in ('recon','object_usd'):
            if sha256(meta['source_files'][key]) != meta['source_hashes'][key]:
                raise ValueError('Source geometry/contact identity changed')
        start = int(row['eligible_start_ranges'][0][0]);span = slice(start,start+128)
        window = {k:v[span] for k,v in params.items()}
        vertices,joints = body.forward(window,rotation,offset)
        target = readonly_array(meta,'motion_world')[span,-66:].reshape(128,22,3)*2
        fk_error = float(np.abs(joints-target).max())
        if fk_error>2e-6:raise ValueError('Contact skin and motion targets disagree')
        ov,of = readonly_array(meta,'scene_object_vertices'),readonly_array(meta,'scene_object_faces')
        sv = np.concatenate((ov,readonly_array(meta,'scene_ground_vertices')))
        sf = np.concatenate((of,readonly_array(meta,'scene_ground_faces')+len(ov)))
        source = sole_measure(vertices,body.feet,sv,sf,support)
        raw = load_pickle(meta['source_files']['recon'])
        probability = contact_probabilities(raw['human_data']['foot_contact_probs'],
            meta['frames'],meta['source_fps'],meta['fps'])[span]
        value = contact_reliability(probability,source['sole_gap_q10_m'],
            source['sole_supported_fraction'],source['foot_speed_mps'],
            readonly_array(meta,'observation_valid')[span],policy)
        centers = np.stack([vertices[:,ids].mean(1) for ids in body.feet],1)
        velocity = torch.tensor(np.diff(centers,axis=0)[None]*30,requires_grad=True)
        finite = np.isfinite(source['sole_gap_q10_m'])
        gap = torch.tensor(np.where(finite,source['sole_gap_q10_m']+.04,0)[None],requires_grad=True)
        weight = torch.tensor(value['weight'][None],requires_grad=True)
        losses = contact_terms(gap,velocity,weight=weight,stance=value['stance'][None],
            distance_valid=finite[None],velocity_valid=np.ones((1,127,2),bool),
            signal=torch.ones(1),policy=policy)
        sum(losses.values()).backward()
        if weight.grad is not None or not bool(torch.isfinite(gap.grad).all()):
            raise ValueError('Reliability gradients or nonfinite geometry gradients')
        label = args.output/(row['category']+'.npz')
        metadata = dict(path=row['path'],split='train',start=start,frames=128,
            metadata_sha256=row['metadata_sha256'],input_sha256=meta['input_sha256'],
            policy_sha256=sha256(args.config),source_evidence_is_model_input=False,
            preparation_scope='acceptance_window_only')
        np.savez_compressed(label,contact_weight=value['weight'],contact_stance=value['stance'],
            metadata_json=json.dumps(metadata))
        weights = value['weight'];stance = value['stance']
        evidence.append(dict(**metadata,category=row['category'],source_fk_max_error_m=fk_error,
            foot_frames=int(weights.size),nominal_stance_foot_frames=int(stance.sum()),
            active_contact_foot_frames=int((weights>0).sum()),
            downweighted_stance_foot_frames=int((stance&(weights>0)&(weights<.99)).sum()),
            masked_stance_foot_frames=int((stance&(weights==0)).sum()),
            swing_weights_zero=bool((weights[~stance]==0).all()),
            independent_left_right_frames=int((weights[:,0]!=weights[:,1]).sum()),
            gradient_to_source_weights=False,geometry_probe_losses={k:float(v.detach()) for k,v in losses.items()},
            sidecar=str(label.resolve()),sidecar_sha256=sha256(label)))
    result = dict(state='passed',scope='component_plus_four_real_train_windows',
        config=str(args.config.resolve()),policy_sha256=sha256(args.config),
        code_sha256={str(p):sha256(ROOT/p) for p in (
            'scenemi/contact_loss.py','scenemi/data/contact_reliability.py')},
        selection_sha256=bundle.contract_sha256 if hasattr(bundle,'contract_sha256') else sha256(ROOT/'configs/scene_data_current.json'),
        source_mesh_used_only_for_label_quality=True,source_evidence_is_model_input=False,
        new_training_started=False,current_training_changed=False,test_arrays_opened=False,
        all_corpus_sidecars_prepared=False,observed_plane_query_integration_complete=False,
        geometry_coefficients_selected=False,training_population=population,cases=evidence,
        completed_at_utc=now())
    atomic_json(args.output/'RESULT.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=DEFAULT_POLICY)
    p.add_argument('--output',type=Path,required=True)
    audit(p.parse_args())


if __name__=='__main__':main()
