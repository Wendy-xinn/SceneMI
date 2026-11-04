"""Sample screened training windows and review their actual scene conditions.

Use the retained source reconstruction viewer; no model generation or motion
correction is performed. Synthetic chairs remain a separately labelled layer.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np

from scenemi.config import ROOT
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.grail_augmentation import observed_prefix
from scenemi.visualization import GrailPlayback
from experiments.offline_camera_retrain_v1.ops.grail_preview import prepare, validate_prepared
from experiments.offline_camera_retrain_v1.ops.viewer_geometry import readonly_array


CATEGORIES = {'curb':'路沿', 'slope':'斜坡', 'stair_p1':'楼梯 p1',
              'stair_p2':'楼梯 p2', 'sitting':'坐下（合成补椅）'}


def choose_windows(rows, seed, per_category=2):
    rng = np.random.default_rng(seed)
    result = []
    for category in CATEGORIES:
        families = defaultdict(list)
        for row in sorted(rows, key=lambda x:x['sequence_id']):
            if row['category']==category:
                if row['split']!='train' or row['eligible_windows']<=0:
                    raise ValueError('Review requires screened training windows')
                families[row['object_family_id']].append(row)
        keys = sorted(families)
        if len(keys)<per_category:
            raise ValueError('Not enough independent source families: '+category)
        for family in rng.choice(keys, per_category, replace=False):
            options = families[family]
            row = dict(options[int(rng.integers(len(options)))])
            lengths = [b-a+1 for a,b in row['eligible_start_ranges']]
            index = int(rng.integers(sum(lengths)))
            for (a,b),length in zip(row['eligible_start_ranges'],lengths):
                if index<length:
                    row['review_start'] = a+index
                    break
                index -= length
            result.append(row)
    return result


def build(output, seed, per_category):
    sources = [ROOT/'runs/grail_ego_quality_v2/full/train/windows.jsonl',
               ROOT/'runs/grail_ego_quality_v2/chair400/SELECTION.jsonl']
    rows = [json.loads(line) for file in sources for line in file.read_text().splitlines()]
    selected = choose_windows(rows, seed, per_category)
    output.mkdir(parents=True, exist_ok=True)
    selection = dict(seed=seed, per_category=per_category, selection='uniform family, recording, eligible start; no visual resampling',
        sources={str(p):sha256(p) for p in sources}, cases=selected)
    path = output/'SELECTION.json'
    if path.exists() and json.loads(path.read_text())!=selection:
        raise ValueError('Existing review selection differs; choose a new output')
    atomic_json(path,selection)
    manifest = output/'SOURCE.jsonl'
    manifest.write_text(''.join(json.dumps(dict(path=r['path'],metadata_sha256=r['metadata_sha256']))+'\n' for r in selected))
    prepared = output/'PREPARED_CASES.json'
    if not prepared.exists():
        prepare(manifest, output)
    payload = json.loads(prepared.read_text())
    prototypes = {r['path']:r for r in json.loads((ROOT/'runs/grail_ego_quality_v2/chair400/conditions/PROTOTYPES.json').read_text())['cases']}
    evidence = []
    for number,(case,row) in enumerate(zip(payload['cases'],selected),1):
        if case['sequence_id']!=row['sequence_id']:
            raise ValueError('Prepared source order differs')
        first = row['review_start']
        if any(a<=first+127 and b>=first for a,b in row['bad_frame_ranges']):
            raise ValueError('Review overlaps excluded frames')
        synthetic = row['category']=='sitting'
        note = ('完整椅子为训练合成增强，未被原相机完整看到；真实历史用青色、合成椅子用橙色。'
                if synthetic else '从 v2 支撑/接触筛选通过的地形训练窗口随机选取。')
        case.update(label=f'{number:02d} · {CATEGORIES[row["category"]]} · 原帧 {first}–{first+127}',
            quality_review=dict(kind='random_retained', start=first, quality=row, note=note))
        folder = Path(row['path'])
        if sha256(folder/'metadata.json')!=row['metadata_sha256']:
            raise ValueError('Source metadata changed')
        meta = dict(json.loads((folder/'metadata.json').read_text()), path=str(folder))
        actual = observed_prefix(readonly_array(meta,'visible_frame_points_scenemi_yup'),
            readonly_array(meta,'visible_frame_owner'),first,.05)
        artificial = np.empty((0,3),np.float32)
        source = None
        if synthetic:
            source = prototypes[row['path']]
            if sha256(source['condition_file'])!=source['condition_sha256']:
                raise ValueError('Synthetic chair points changed')
            with np.load(source['condition_file'],allow_pickle=False) as arrays:
                artificial = arrays['synthetic_chair_points_m']
        cloud = output/f'condition_{number:02d}.npz'
        np.savez_compressed(cloud,actual=actual,synthetic=artificial)
        case['review_condition'] = dict(path=str(cloud.resolve()),sha256=sha256(cloud),
            actual_points=len(actual),synthetic_points=len(artificial),history_cutoff=first,
            synthetic_provenance=source,body_or_camera_modified=False)
        evidence.append(dict(sequence_id=row['sequence_id'],category=row['category'],start=first,
            screened=True,synthetic_chair=synthetic,**case['review_condition']))
    atomic_json(prepared,payload)
    validate_prepared(prepared)
    atomic_json(output/'AUDIT.json',dict(state='passed',seed=seed,cases=evidence,
        model_inference=False,source_motion_corrected=False,physics_validity_proven=False,
        implementation_sha256=sha256(__file__)))
    return prepared


class ReviewPlayback(GrailPlayback):
    def __init__(self,cases,port):
        self.review_layers = []
        self.show_actual = self.show_synthetic = None
        super().__init__(cases,port)
        self.show_actual = self.server.gui.add_checkbox('窗口起点已观测历史（青色）',initial_value=True)
        self.show_synthetic = self.server.gui.add_checkbox('合成补全椅子（橙色，训练增强）',initial_value=True)
        for control in (self.show_actual,self.show_synthetic):
            control.on_update(lambda _:self.update())
        self.server.gui.add_markdown('**这批是原始数据检查，不是模型预测。** 每类两个不同物体家族；'
            '坐下来自选定 400 条增强录制。灰色完整源场景仅作参照，橙色椅子是合成条件。'
            '青色仅累计到选定窗口起点，逐帧可见点沿用原页面图层。')
        self.load_review_layers()

    def load_review_layers(self):
        for handle in self.review_layers:handle.remove()
        self.review_layers = []
        record = self.cases[self.index]['review_condition']
        if sha256(record['path'])!=record['sha256']:
            raise ValueError('Review condition changed')
        with np.load(record['path'],allow_pickle=False) as arrays:
            for key,color in [('actual',(30,175,185)),('synthetic',(245,150,25))]:
                handle = self.points('/ego/review/'+key,color,.012)
                handle.points = arrays[key]
                self.review_layers.append(handle)
        self.update()

    def load_case(self):
        # MotionViewer removes tracked handles before changing GUI sliders;
        # those slider callbacks must not touch the preceding case's handles.
        self.review_layers = []
        super().load_case()
        if self.show_actual is not None:self.load_review_layers()

    def update(self):
        super().update()
        if self.loading:return
        if self.show_actual is not None and len(self.review_layers)==2:
            active = bool(self.panels['ego']['enabled'].value)
            self.review_layers[0].visible = active and bool(self.show_actual.value)
            self.review_layers[1].visible = active and bool(self.show_synthetic.value)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','serve'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=2026102302)
    p.add_argument('--per-category',type=int,default=2)
    p.add_argument('--port',type=int,default=8840)
    a=p.parse_args()
    if a.action=='prepare':print(build(a.output.resolve(),a.seed,a.per_category))
    else:
        prepared=a.output.resolve()/'PREPARED_CASES.json'
        viewer=ReviewPlayback(validate_prepared(prepared),a.port)
        atomic_json(a.output/'SERVER.json',dict(pid=os.getpid(),url=f'http://localhost:{a.port}/',
            manifest=str(prepared),data_only=True,model_inference=False,case_count=len(viewer.cases)))
        viewer.run()


if __name__=='__main__':main()
