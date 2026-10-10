"""Select fresh confirmation recordings using identities only, never scores."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE=Path(__file__).parent;BASE=HERE/'runs';OUT=BASE/'state_relative_spline_oct10'
SOURCE=Path('/home/wenxin/projects/diffusion-motion-inbetweening/experiments/offline_sequence_v1/data')


def main():
    old=json.loads((BASE/'bounded_head_adaptation_oct10/holdout_manifest.json').read_text())
    dev=json.loads((BASE/'body_history_replan_oct10/screen_protocol.json').read_text())
    excluded=set(old['excluded_development_sequences'])|{r['sequence_id'] for r in old['selected']}|{r['sequence_id'] for r in dev['selected']}
    records={r['sequence_id']:r['recording'] for r in map(json.loads,(SOURCE/'egobody_sequences/validation.jsonl').read_text().splitlines())}
    excluded_records={records[n] for n in excluded if n in records}
    candidates={group:defaultdict(list) for group in ('trumans','camera_wearer','interactee','rich')}
    for row in map(json.loads,(BASE/'native_dynamic_scene20_contact_55k_oct07/full_validation/rows.jsonl').read_text().splitlines()):
        w=row['window'];name=w['sequence_id']
        if w['length']!=128 or name in excluded or records.get(name) in excluded_records:continue
        candidates[w['group']][name].append((w,row['identity']['source_start_30fps']))
    rng=np.random.default_rng(2026101031);selected=[]
    for group in candidates:
        names=list(rng.permutation(sorted(candidates[group])));chosen=[];seen=set()
        for name in names:
            record=records.get(name,name)
            if record in seen or len(candidates[group][name])<3:continue
            chosen.append(name);seen.add(record)
            if len(chosen)==4:break
        assert len(chosen)==4,(group,len(chosen))
        for name in chosen:
            windows=sorted(candidates[group][name],key=lambda pair:pair[1])
            for block in np.array_split(np.arange(len(windows)),3):
                w,time=windows[int(rng.choice(block))];selected.append(dict(w,expected_source_start_30fps=time,holdout_recording=records.get(name,name)))
    assert len(selected)==48
    out=dict(status='reserved_before_prediction',seed=2026101031,excluded_sequences=sorted(excluded),excluded_recordings=sorted(excluded_records),selected=selected,windows=48,sequences=16,recordings=len({r['holdout_recording'] for r in selected}),scope='fresh for current refinement selection; official validation already evaluated by original55k, not independent external test; no score-based selection')
    OUT.mkdir(exist_ok=True);(OUT/'confirmation_manifest.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ('selected','excluded_sequences','excluded_recordings')}))

if __name__=='__main__':main()
