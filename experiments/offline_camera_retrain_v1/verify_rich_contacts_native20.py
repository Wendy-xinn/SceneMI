"""Check packed native labels against released annotations and missing colors."""
import json,pickle
from pathlib import Path
import numpy as np
from experiments.offline_camera_retrain_v1.export_rich_contact_native20 import read_smplx,RAW,HERE
root=HERE/'data/rich_contact_native20_v1'
rows=[json.loads(l) for l in (root/'manifest.jsonl').read_text().splitlines()]
report=[]
for row in rows:
 p=root/row['split']/row['sequence_id']/'contacts.npz'
 with np.load(p) as a:
  assert a['smpl_packed'].shape==(row['frames'],862)
  assert a['smplx_packed'].shape==(row['frames'],1310)
  assert np.max(np.abs(a['label_frame_ids']-a['source_frame_ids']))<=.5+1e-6
  assert int(a['valid'][:,0].sum())==row['smpl_labeled']
  for t in sorted({0,row['frames']//2,row['frames']-1}):
   frame=int(a['label_frame_ids'][t]);subject=row['sequence_id'].split('_')[1];src=RAW/f'{row["split"]}_hsc'/row['sequence_id']/f'{frame:05d}'/f'{subject}.pkl'
   if a['valid'][t,0]:
    truth=pickle.load(src.open('rb'),encoding='latin1')['contact'].astype(bool)
    assert np.array_equal(np.unpackbits(a['smpl_packed'][t])[:6890],truth)
   if a['valid'][t,1]:assert np.array_equal(np.unpackbits(a['smplx_packed'][t])[:10475],read_smplx(src.with_suffix('.obj')))
 report.append(dict(sequence=row['sequence_id'],checked_frames=3))
# Existing uncolored mesh must remain missing, not a negative label.
assert read_smplx(RAW/'train_hsc/ParkingLot1_002_burpee3/00283/002.obj') is None
summary=dict(status='passed',sequences=len(rows),frames=sum(r['frames'] for r in rows),smpl_labeled=sum(r['smpl_labeled'] for r in rows),smplx_labeled=sum(r['smplx_labeled'] for r in rows),missing_is_unknown=True,original_split_preserved=True,model_contact_head_enabled=False,checks=report)
(root/'verification.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='checks'},indent=2))
