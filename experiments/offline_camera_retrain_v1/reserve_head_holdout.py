"""Reserve new validation recordings/sequences without consulting prediction scores."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

BASE=Path(__file__).parent/'runs';OUT=BASE/'bounded_head_adaptation_oct10'
SOURCE=Path('/home/wenxin/projects/diffusion-motion-inbetweening/experiments/offline_sequence_v1/data')

def main():
    dev=json.loads((BASE/'turn_balance_oct09/manifest.json').read_text())
    excluded={r['sequence_id'] for r in dev['selected']}
    excluded|={r['row']['identity']['sequence_id'] for r in json.loads((BASE/'native_dynamic_scene20_contact_55k_oct07/scene_effect_demo/manifest.json').read_text())['cases']}
    records={r['sequence_id']:r['recording'] for r in map(json.loads,(SOURCE/'egobody_sequences/validation.jsonl').read_text().splitlines())}
    excluded_records={records[name] for name in excluded if name in records}
    candidates={group:defaultdict(list) for group in ('trumans','camera_wearer','interactee','rich')}
    for row in map(json.loads,(BASE/'native_dynamic_scene20_contact_55k_oct07/full_validation/rows.jsonl').read_text().splitlines()):
        w=row['window'];name=w['sequence_id']
        if w['length']!=128 or name in excluded or records.get(name) in excluded_records:continue
        candidates[w['group']][name].append((w,row['identity']['source_start_30fps']))
    rng=np.random.default_rng(2026101017);selected=[]
    for group in candidates:
        names=list(rng.permutation(sorted(candidates[group])));chosen=[];seen_records=set()
        for name in names:
            recording=records.get(name,name)
            if recording in seen_records:continue
            chosen.append(name);seen_records.add(recording)
            if len(chosen)==4:break
        assert len(chosen)==4,(group,len(chosen))
        for name in chosen:
            windows=sorted(candidates[group][name],key=lambda pair:pair[1])
            assert len(windows)>=3
            for block in np.array_split(np.arange(len(windows)),3):
                w,time=windows[int(rng.choice(block))];selected.append(dict(w,expected_source_start_30fps=time,holdout_recording=records.get(name,name)))
    assert len(selected)==48
    assert not ({r['sequence_id'] for r in selected}&excluded)
    assert not ({records[r['sequence_id']] for r in selected if r['sequence_id'] in records}&excluded_records)
    out=dict(status='reserved_not_evaluated',seed=2026101017,selection='four validation sequences per role, distinct recordings within each EgoBody role; three random temporal strata; no prediction-score selection',excluded_development_sequences=sorted(excluded),excluded_egobody_recordings=sorted(excluded_records),windows=48,sequences=16,recordings=len({r['holdout_recording'] for r in selected}),scope='unused by recent 192-window short-trial selection; official validation, not a pristine test set; RICH scenes/subjects may overlap official training',selected=selected)
    (OUT/'holdout_manifest.json').write_text(json.dumps(out,indent=2));print('reserved',out['windows'],'windows',out['sequences'],'sequences',out['recordings'],'recordings')

if __name__=='__main__':main()
