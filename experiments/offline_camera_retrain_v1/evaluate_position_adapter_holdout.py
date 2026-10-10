"""Independent validation sequences reserved before the adapter trial."""
import json
from pathlib import Path
from experiments.offline_camera_retrain_v1 import evaluate_position_adapter as evaluation

def main():
 root=Path(__file__).parent/'runs/position_adapter_head_oct10'
 manifest=json.loads((root.parent/'bounded_head_adaptation_oct10/holdout_manifest.json').read_text())
 out=root/'holdout';reference=root/'holdout_reference'
 out.mkdir(exist_ok=True);reference.mkdir(exist_ok=True)
 (reference/'manifest.json').write_text(json.dumps(manifest,indent=2))
 if not (out/'last.pt').exists():(out/'last.pt').symlink_to((root/'last.pt').resolve())
 evaluation.OUT=out;evaluation.REFERENCE=reference
 evaluation.main()
 # Explicitly record consumption; reservation file stays immutable.
 (out/'panel_use.json').write_text(json.dumps({'status':'evaluated','purpose':'confirmation only after development screen passed','previous_full55k_validation':True,'pristine_test':False},indent=2))

if __name__=='__main__':main()
