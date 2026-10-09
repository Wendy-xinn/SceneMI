"""Add metrics to saved full-validation joints without regenerating predictions."""
import argparse,json,time
from pathlib import Path
import numpy as np
from experiments.offline_camera_retrain_v1.standard_motion_metrics import standard_motion_metrics,METRIC_PROTOCOL
from experiments.offline_camera_retrain_v1.evaluate_report_oct07 import aggregate

def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);a=p.parse_args()
    start=time.monotonic();rows=[];output=a.evaluation/'standard_rows.jsonl'
    with output.open('w') as stream:
        for line in (a.evaluation/'rows.jsonl').open():
            row=json.loads(line)
            with np.load(a.evaluation/row['motion_file']) as motion:
                for variant in ('scene_on','scene_off'):
                    values=standard_motion_metrics(motion[variant],motion['truth'])
                    if abs(values['w_mpjpe_mm']-row[variant]['mpjpe_cm']*10)>1e-3:
                        raise ValueError('Saved joints disagree with original world MPJPE')
                    row[variant].update(values)
            rows.append(row);stream.write(json.dumps(row)+'\n')
    original=json.loads((a.evaluation/'metrics.json').read_text())
    result={'status':'completed','full_coverage':original['full_coverage'],'windows':len(rows),'reused_saved_predictions':True,'elapsed_s':time.monotonic()-start,'metric_protocol':METRIC_PROTOCOL,'summary':aggregate(rows)}
    assert len(rows)==sum(1 for _ in (a.evaluation/'rows.jsonl').open())
    (a.evaluation/'standard_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'windows':len(rows),'elapsed_s':result['elapsed_s']}))
if __name__=='__main__':main()
