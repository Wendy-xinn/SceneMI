"""Behavioral checks: memory spans windows, future is absent, motion is not a trail."""
import json
import numpy as np
from experiments.offline_camera_retrain_v1.training_contract_v2 import CausalSceneMemory,domain_probabilities,validate_contract
m=CausalSceneMemory();anchors=np.zeros((1,3),np.float32)
x,valid=m.bps(np.zeros(3),np.eye(3),anchors);assert not valid.any()
m.observe(0,[[0,0,2],[1,0,0],[99,0,0]],[0,1,100])
old=m.bps(np.zeros(3),np.eye(3),anchors)[0].copy()
# Advancing into another target window must retain earlier static surfaces.
m.observe(20,[[0,0,3],[2,0,0]],[0,1]);static,dynamic=m.surfaces()
assert np.array_equal(dynamic,[[2,0,0]]) and len(static)==2
assert any(np.array_equal(p,[0,0,2]) for p in static)
# Later observation cannot mutate the already saved earlier condition.
assert np.array_equal(old,[[.5,0,0]])
m.observe(21,np.empty((0,3)),np.empty(0,int));assert len(m.surfaces()[1])==0
try:m.observe(19,[[0,0,0]],[0]);raise AssertionError('Backwards time accepted')
except ValueError:pass
m.reset();assert not len(m.surfaces()[0])
for sizes in [{'large':100,'small':1},{'a':3,'b':3}]:
    p=domain_probabilities(sizes,1);assert np.isclose(sum(p.values()),1)
    assert np.isclose(p[next(iter(sizes))],next(iter(sizes.values()))/sum(sizes.values()))
try:validate_contract({});raise AssertionError('Invalid historical contract accepted')
except ValueError:pass
print(json.dumps(dict(status='passed',checks=['unknown empty map','body excluded','static history across windows','current dynamic without trail','unobserved dynamic removed','time order','recording reset','duration sampling','reject unqualified contract'])))
