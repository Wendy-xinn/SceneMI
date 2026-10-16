import json,subprocess,time,traceback
from pathlib import Path
base=Path("experiments/offline_camera_retrain_v1/runs/turn_balance_oct09")
status=base/"recovery_status.json"
try:
 status.write_text(json.dumps({"status":"training_hard","started_unix":time.time()}))
 with (base/"hard_training.log").open("w") as log:
  p=subprocess.Popen(json.loads((base/"hard_command.json").read_text()),stdout=log,stderr=subprocess.STDOUT)
  (base/"hard_training.pid").write_text(str(p.pid))
  if p.wait():raise RuntimeError("Hard-head training failed")
 status.write_text(json.dumps({"status":"evaluating"}))
 with (base/"evaluation.log").open("w") as log:
  subprocess.run(["/home/wenxin/miniconda3/envs/scenemi/bin/python","-u","-m","experiments.offline_camera_retrain_v1.evaluate_turn_balance"],stdout=log,stderr=subprocess.STDOUT,check=True)
 status.write_text(json.dumps({"status":"completed"}))
except Exception as exc:
 status.write_text(json.dumps({"status":"failed","error":str(exc)}));raise
