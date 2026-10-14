"""Full official TRAIN sampling/VAL plan audit; zero optimizer updates."""
from collections import Counter
import datetime as dt
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.prepare_native30 import read_config,write_json
from experiments.offline_camera_retrain_v1.native30_overnight import scoped_data
from experiments.offline_camera_retrain_v1.native30_sampling import StratifiedRecordingSampler,recording_key
from experiments.offline_camera_retrain_v1.native30_validation import monitoring_plan


def main():
    config=read_config(HERE/'native30_recording_headonly_config.json')
    data=scoped_data(config,'train');plan=data.sampling_plan(config['ego_probability'])
    sampler=StratifiedRecordingSampler(data,plan,config['sampler'],config['seed']+371)
    counts=Counter();records={};cooldown=overlap=0;blocks=[];began=time.monotonic()
    for i in range(6000):
        draw=sampler.draw();counts[draw['group']+'/'+draw['mode']]+=1
        records.setdefault(draw['group']+'/'+draw['mode'],Counter())[draw['recording_key']]+=1
        cooldown+=draw['cooldown_fallback'];overlap+=draw['overlap_fallback'];blocks.append(draw['mode'])
        if len(blocks)==10:
            assert Counter(blocks)==dict(ego=7,exo=3);blocks=[]
    sampling_s=time.monotonic()-began
    state=sampler.state_dict();next_draws=[sampler.draw() for _ in range(100)]
    restored=StratifiedRecordingSampler(data,plan,config['sampler'],1);restored.load_state_dict(state)
    assert next_draws==[restored.draw() for _ in range(100)]
    coverage={}
    for bucket,pool in sampler.pools.items():
        actual=records[bucket];windows=sum(len(v) for v in pool.values())
        coverage[bucket]=dict(eligible_recordings=len(pool),drawn_recordings=len(actual),
            minimum_draws=min(actual.get(r,0) for r in pool),maximum_draws=max(actual.get(r,0) for r in pool),
            old_largest_recording_probability_within_bucket=max(len(v) for v in pool.values())/windows,
            new_recording_probability_within_bucket=1/len(pool))
        assert coverage[bucket]['maximum_draws']-coverage[bucket]['minimum_draws']<=1
    val=scoped_data(config,'validation');panels=[monitoring_plan(val,config,s) for s in (1,1000,2000)]
    fixed=lambda rows:[(r['group'],r['mode'],r['sequence_id'],r['start'],r['case_seed']) for r in rows if r['panel']=='fixed']
    assert fixed(panels[0])==fixed(panels[1])==fixed(panels[2])
    run=HERE/'runs/native30_scene_relations_oct14/experiment'
    identities=json.loads((run/'identities.json').read_text())['train'];pilot=Counter()
    for line in (run/'candidate/trace.jsonl').read_text().splitlines():
        for i in json.loads(line)['identities']:
            m=identities[i];pilot[('egobody' if m['group'] in ('camera_wearer','interactee') else m['group'])+'/'+m['mode']]+=1
    corpus={c:sum(r['probability'] for r in plan if ('egobody' if r['group'] in ('camera_wearer','interactee') else r['group'])==c)
            for c in ('trumans','egobody','rich')}
    report=dict(status='passed',optimizer_steps=0,created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        config=config,train_source_manifest=str(data.manifest),plan=plan,combined_corpus_probabilities=corpus,
        draws=6000,counts=dict(counts),recording_coverage=coverage,exact_mode_blocks=True,
        cooldown_fallbacks=cooldown,overlap_fallbacks=overlap,sampling_only_s=sampling_s,
        resume_next_100_draws_identical=True,previous_pilot_400_draws=dict(pilot),
        validation_panels=panels,default_monitor_prediction_calls=24,scene_check_prediction_calls=36,
        scope='Actual complete official TRAIN index and VAL plans. Sampling/head-only monitoring only; no new quality metrics or complete preparation claimed.')
    write_json(Path(__file__).with_name('SAMPLING_MONITOR_AUDIT.json'),report)
    print(json.dumps(dict(status=report['status'],counts=report['counts'],corpus=corpus,
        cooldown_fallbacks=cooldown,overlap_fallbacks=overlap,sampling_only_s=sampling_s),indent=2),flush=True)


if __name__=='__main__':main()
