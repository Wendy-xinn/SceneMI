"""Strict, explicit warm-start gate for reconstructed native archive provenance."""
import json
from pathlib import Path
from experiments.offline_camera_retrain_v1.data import HERE,SOURCE

PIPELINE=['data.py','native_body_data.py','scene_model.py','temporal_windows.py','causal_scene.py',
          'recover_native_source_archive.py','audit_recovered_native_source.py','source_recovery_contract.py']

def validate_source_recovery(path,reference_source_hash,args):
 from experiments.offline_camera_retrain_v1.train import audited_data_fingerprint,fingerprint
 report=json.loads(Path(path).read_text());ready=json.loads((SOURCE/'READY.json').read_text())
 if args.body_protocol!='native_v1' or args.trumans_window_protocol!='temporal_valid_v1' or not args.temporal_scene_manifest or not args.trumans_scene_manifest:
  raise ValueError('Recovery transfer requires complete native exact20 scene inputs')
 if report.get('status')!='passed' or report.get('protocol')!='native-source-recovery-v1' or report.get('windows')!=192 or report.get('paired_predictions')!=384:
  raise ValueError('Recovery replay audit incomplete')
 if report.get('reference_source_hash')!=reference_source_hash:raise ValueError('Recovery reference checkpoint/source mismatch')
 if max(report.get('maximum_difference',{'failed':float('inf')}).values())>1e-5:raise ValueError('Recovered native tensors/predictions differ')
 if report.get('pipeline_sha256')!=fingerprint([HERE/name for name in PIPELINE]):raise ValueError('Recovery pipeline changed after replay audit')
 if report.get('audited_archive_fingerprint')!=audited_data_fingerprint('native20_faceout_oct07'):raise ValueError('Recovered archive changed after replay audit')
 if ready.get('reconstruction_protocol')!=report['protocol']:raise ValueError('Recovery READY mismatch')
 if Path(ready['semantic_audit_path']).resolve()!=Path(path).resolve():raise ValueError('Unexpected recovery audit file')
 return report
