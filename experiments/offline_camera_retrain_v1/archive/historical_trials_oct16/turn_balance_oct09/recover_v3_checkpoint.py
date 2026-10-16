import json,subprocess
from pathlib import Path
b=Path("experiments/offline_camera_retrain_v1/runs/turn_balance_oct09")
s=b/"recovery_status.json"
try:
 s.write_text(json.dumps({"status":"rerunning_v3_after_corrupt_checkpoint"}))
 with (b/"training_rerun.log").open("w") as log:subprocess.run(json.loads((b/"command.json").read_text()),stdout=log,stderr=subprocess.STDOUT,check=True)
 s.write_text(json.dumps({"status":"evaluating"}))
 with (b/"evaluation.log").open("w") as log:subprocess.run(["/home/wenxin/miniconda3/envs/scenemi/bin/python","-u","-m","experiments.offline_camera_retrain_v1.evaluate_turn_balance"],stdout=log,stderr=subprocess.STDOUT,check=True)
 subprocess.run(["/home/wenxin/miniconda3/envs/scenemi/bin/python","-m","experiments.offline_camera_retrain_v1.summarize_turn_balance"],check=True)
 s.write_text(json.dumps({"status":"completed"}))
except Exception as e:s.write_text(json.dumps({"status":"failed","error":str(e)}));raise
