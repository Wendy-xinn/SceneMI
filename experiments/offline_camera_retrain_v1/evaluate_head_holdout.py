"""Apply frozen candidates to the pre-reserved official validation panel."""
import json
from pathlib import Path
from experiments.offline_camera_retrain_v1 import evaluate_bounded_head as evaluation


def main():
    root=Path(__file__).parent/'runs/bounded_head_adaptation_oct10'
    manifest=json.loads((root/'holdout_manifest.json').read_text())
    out=root/'holdout';reference=root/'holdout_reference'
    out.mkdir(exist_ok=True);reference.mkdir(exist_ok=True)
    (reference/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for label in ('joint_all','joint_staged','rotation_only','mild_only'):
        alias=out/label
        if not alias.exists():alias.symlink_to((root/label).resolve(),target_is_directory=True)
    evaluation.OUT=out;evaluation.REFERENCE=reference
    evaluation.main()

if __name__=='__main__':main()
