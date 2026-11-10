"""Frozen scene-validation coverage and retained checkpoint candidates.

Quick validation covers every native recording and a stratified GRAIL sample.
Comprehensive validation covers every family and includes the quick windows.
Neither stage opens test data or synthesizes unseen scene observations.
"""
from collections import Counter, defaultdict
import os
from pathlib import Path

import numpy as np

from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.evaluation import summarize


def validation_manifests(data, seed, grail_per_category=20):
    if data.split != 'validation' or grail_per_category <= 0:
        raise ValueError('Expected canonical validation data and a positive sampling budget')
    rng = np.random.default_rng(seed)
    categories, quick, comprehensive = defaultdict(list), [], []
    by_family = {}
    for family in data.families['grail']:
        members = data.groups['grail'][family]
        labels = {data.metadata(data.records[m])['category'] for m in members}
        if len(labels) != 1:
            raise ValueError('A GRAIL object family spans multiple action categories')
        category = next(iter(labels))
        categories[category].append(family)
        by_family[family] = category
    if set(categories) != {'curb', 'slope', 'sitting', 'stair_p1', 'stair_p2'}:
        raise ValueError('All five GRAIL action categories must be present')
    selected = set()
    for category in sorted(categories):
        families = sorted(categories[category])
        selected.update(rng.choice(families, min(grail_per_category, len(families)), replace=False))

    def case(member, start):
        r = data.records[member]
        return dict(member=int(member), start=int(start), dataset=r['dataset'], family=r['family'],
                    category=by_family[r['family']] if r['dataset']=='grail' else r['dataset'])

    # All native recordings, including the available EgoBody sensor profiles.
    for name in ('egobody', 'trumans'):
        for family in data.families[name]:
            for member in data.groups[name][family]:
                starts = data.records[member]['starts']
                first = int(starts[len(starts)//2])
                quick.append(case(member, first))
                comprehensive.append(case(member, first))
                # Prefer a distant window; short recordings may have only one.
                other = int(starts[0] if first-starts[0] >= starts[-1]-first else starts[-1])
                if other != first:
                    comprehensive.append(case(member, other))
    for family in data.families['grail']:
        members = data.groups['grail'][family]
        order = rng.permutation(members).tolist()
        member = order[0]
        starts = data.records[member]['starts']
        first = int(starts[len(starts)//2])
        initial = case(member, first)
        comprehensive.append(initial)
        if family in selected:
            quick.append(initial)
        if len(order)>1:
            member = order[1]
            starts = data.records[member]['starts']
            comprehensive.append(case(member, int(starts[len(starts)//2])))
        elif int(starts[0]) != first or int(starts[-1]) != first:
            other = int(starts[0] if first-starts[0] >= starts[-1]-first else starts[-1])
            comprehensive.append(case(member, other))
    key = lambda row:(row['member'],row['start'])
    if len(set(map(key, quick)))!=len(quick) or len(set(map(key, comprehensive)))!=len(comprehensive):
        raise ValueError('Duplicate validation windows')
    if not set(map(key,quick)).issubset(set(map(key,comprehensive))):
        raise ValueError('Quick validation must be nested in comprehensive validation')
    coverage = {tier:dict(windows=len(cases),
        per_dataset=dict(Counter(r['dataset'] for r in cases)),
        families={name:len({r['family'] for r in cases if r['dataset']==name})
                  for name in ('egobody','trumans','grail')},
        grail_category_windows=dict(Counter(r['category'] for r in cases if r['dataset']=='grail')))
        for tier,cases in (('quick',quick),('comprehensive',comprehensive))}
    return dict(quick=quick,comprehensive=comprehensive,coverage=coverage,
        scope='All validation families, sampled windows; not every validation frame.',
        synthetic_chairs=False,test_used=False)


def aggregate_scene_rows(rows):
    groups = {name:summarize([r for r in rows if r['identity']['dataset']==name])
              for name in sorted({r['identity']['dataset'] for r in rows})}
    categories = {name:summarize([r for r in rows if r['identity']['dataset']=='grail'
                                  and r['identity']['category']==name])
                  for name in sorted({r['identity']['category'] for r in rows
                                      if r['identity']['dataset']=='grail'})}
    # Give GRAIL's five actions equal weight instead of letting its many stair
    # objects overwhelm sitting, slopes and curbs. Other corpora use family means.
    scores = [g['family_macro_mean']['mpjpe_mm'] for name,g in groups.items() if name!='grail']
    if categories:
        scores.append(float(np.mean([g['family_macro_mean']['mpjpe_mm'] for g in categories.values()])))
    return dict(selection_score_mm=float(np.mean(scores)), per_dataset=groups,
                grail_per_category=categories, rows=rows)


def capture_candidate(checkpoint, selection, step, contract_sha256):
    folder = Path(selection)/'candidates'; folder.mkdir(parents=True,exist_ok=True)
    target = folder/f'step_{step:06d}.pt'
    digest = sha256(checkpoint)
    if target.exists():
        if sha256(target)!=digest:
            raise ValueError('A retained candidate at this step changed')
    else:
        temporary = target.with_suffix('.tmp'); temporary.unlink(missing_ok=True)
        os.link(checkpoint,temporary);os.replace(temporary,target)
    atomic_json(target.with_suffix('.json'),dict(step=step,checkpoint_sha256=digest,
        contract_sha256=contract_sha256,checkpoint=str(target),resumable=True))
    return target


def finalists(quick_results, final_step):
    if len({r['step'] for r in quick_results})!=len(quick_results):
        raise ValueError('Duplicate checkpoint scores')
    if final_step not in {r['step'] for r in quick_results}:
        raise ValueError('The final checkpoint must be evaluated')
    ranked = sorted(quick_results,key=lambda r:(r['selection_score_mm'],r['step']))
    return sorted({r['step'] for r in ranked[:3]}|{final_step})
