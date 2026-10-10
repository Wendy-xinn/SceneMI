"""Reserved validation panel, used only after paired development screen."""
import json
from pathlib import Path
from experiments.offline_camera_retrain_v1 import evaluate_soft_coupled_contact as evaluation

def main():
 root=Path(__file__).parent/'runs/soft_coupled_contact_oct10'
 manifest=json.loads((root.parent/'bounded_head_adaptation_oct10/holdout_manifest.json').read_text())
 out=root/'holdout';reference=root/'holdout_reference';out.mkdir(exist_ok=True);reference.mkdir(exist_ok=True)
 (reference/'manifest.json').write_text(json.dumps(manifest,indent=2))
 for label in ('baseline','coupled','coupled_contact'):
  if not (out/label).exists():(out/label).symlink_to((root/label).resolve(),target_is_directory=True)
 evaluation.OUT=out;evaluation.REFERENCE=reference;evaluation.main()
 (out/'panel_use.json').write_text(json.dumps({'status':'evaluated','prior_full55k_validation':True,'pristine_test':False,'windows':48,'recordings':14},indent=2))
if __name__=='__main__':main()
