"""Native shape decoders sharing the AMASS prior's pelvis reference.

Only the output body-model format is configured. Human coefficients and the
neutral/female/male template are predicted from observations, never supplied
from a target's metadata. GRAIL G1 is a public fixed output template, with its
scale already baked into the vertices. Licensed full meshes stay local.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scenemi.config import ROOT, load_paths
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.geometry.body import PARENTS
from scenemi.geometry.shape import SmplhShape
from scenemi.training import ensure_bounded, now

FORMATS = ('smplh', 'smpl', 'smplx', 'grail_g1')
GENDERS = ('neutral', 'female', 'male')


def regressed_basis(model, eye_ids):
    """Regress joints and head-local eye offsets before discarding the mesh."""
    if tuple(int(v) for v in model.parents[:22]) != PARENTS:
        raise ValueError('Native format does not share the 22-joint body hierarchy')
    with torch.no_grad():
        vertices = model.v_template.detach().cpu().float()
        directions = model.shapedirs[:, :, :10].detach().cpu().float()
        regressor = model.J_regressor[:22].detach().cpu().float()
        rest = regressor @ vertices
        rest_dirs = torch.einsum('jv,vci->jci', regressor, directions)
        eye = vertices[eye_ids].mean(0) - rest[15]
        eye_dirs = directions[eye_ids].mean(0) - rest_dirs[15]
    return dict(rest_template=rest, rest_directions=rest_dirs,
                eye_template=eye, eye_directions=eye_dirs)


def load_native_bases():
    # The frozen loader contains NumPy/chumpy compatibility for local SMPL.
    from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
    from experiments.offline_camera_retrain_v1.ops.grail_source import body_model_factory
    bases = {}
    for kind in ('smpl', 'smplx'):
        per_gender = []
        for gender in GENDERS:
            model = load_model(kind, gender)
            per_gender.append(regressed_basis(model, [2802, 6262] if kind=='smpl' else [9929, 9448]))
            del model
            gc.collect()
        bases[kind] = {key: torch.stack([b[key] for b in per_gender]) for key in per_gender[0]}
    model = body_model_factory('g1')
    bases['grail_g1'] = {key: value[None] for key, value in regressed_basis(model, [9929, 9448]).items()}
    del model
    return bases


class NativeShapeBank(nn.Module):
    """Exact native joint/eye regression, without retaining full body meshes."""
    def __init__(self, common_pelvis, bases=None):
        super().__init__()
        self.register_buffer('common_pelvis', torch.as_tensor(common_pelvis).float().reshape(1, 3))
        for kind, basis in (load_native_bases() if bases is None else bases).items():
            for key, value in basis.items():
                self.register_buffer(f'{kind}_{key}', torch.as_tensor(value).float())

    def forward(self, kind, coefficients, template_index):
        if kind not in FORMATS[1:]:
            raise ValueError('SMPL-H keeps its original decoder')
        if coefficients.ndim != 2 or coefficients.shape[1] != 10 or not torch.isfinite(coefficients).all():
            raise ValueError('Native human decoders require ten finite coefficients')
        if template_index.shape != coefficients.shape[:1]:
            raise ValueError('Expected one predicted template index per window')
        template = getattr(self, f'{kind}_rest_template')[template_index]
        directions = getattr(self, f'{kind}_rest_directions')[template_index]
        eye = getattr(self, f'{kind}_eye_template')[template_index]
        eye_dirs = getattr(self, f'{kind}_eye_directions')[template_index]
        native = template + torch.einsum('bjci,bi->bjc', directions, coefficients)
        rest = native - native[:, :1] + self.common_pelvis
        mount = eye + torch.einsum('bci,bi->bc', eye_dirs, coefficients) + eye.new_tensor([0., 0., .02])
        return dict(rest=rest, native_rest=native, head_camera_translation=mount)


class NativeBodyAdapter(nn.Module):
    """Separate native coefficient heads on the original observation features.

    Template classification is supervised during training; argmax prediction
    is used in both training and inference. There is no target-template input
    and no copying/truncating SMPL-H coefficients into another shape space.
    """
    def __init__(self, common_pelvis, bases=None):
        super().__init__()
        self.bank = NativeShapeBank(common_pelvis, bases)
        self.coefficients = nn.ModuleDict({kind: nn.Linear(256, 10) for kind in ('smpl', 'smplx')})
        self.templates = nn.ModuleDict({kind: nn.Linear(256, 3) for kind in ('smpl', 'smplx')})
        for head in list(self.coefficients.values()) + list(self.templates.values()):
            nn.init.normal_(head.weight, std=.001)
            nn.init.zeros_(head.bias)

    def forward(self, features, formats):
        if formats.shape != features.shape[:1] or formats.dtype != torch.long:
            raise ValueError('Expected a configured native format per window')
        if not ((formats >= 1) & (formats < len(FORMATS))).all():
            raise ValueError('Use the original SMPL-H decoder for AMASS')
        b = len(features)
        output = dict(rest=features.new_zeros(b, 22, 3), native_rest=features.new_zeros(b, 22, 3),
            head_camera_translation=features.new_zeros(b, 3), predicted_native_betas=features.new_zeros(b, 10),
            template_logits=features.new_zeros(b, 3), predicted_template=torch.zeros(b, device=features.device, dtype=torch.long))
        for code, kind in enumerate(FORMATS[1:], 1):
            active = formats == code
            if not active.any():
                continue
            if kind == 'grail_g1':
                coefficients = features.new_zeros(int(active.sum()), 10)
                logits = features.new_zeros(len(coefficients), 3)
                template = torch.zeros(len(coefficients), device=features.device, dtype=torch.long)
            else:
                coefficients = self.coefficients[kind](features[active])
                logits = self.templates[kind](features[active])
                template = logits.argmax(-1)
            shape = self.bank(kind, coefficients, template)
            for key, value in shape.items():
                output[key][active] = value
            output['predicted_native_betas'][active] = coefficients
            output['template_logits'][active] = logits
            output['predicted_template'][active] = template
        return output


def source_assets(paths):
    assets = [paths['smplh_models']/'neutral/model.npz']
    assets += [ROOT.parent/'ProtoMotions/data/smpl'/f'SMPL_{gender.upper()}.pkl' for gender in GENDERS]
    assets += [ROOT/'body_models/smplx'/f'SMPLX_{gender.upper()}.npz' for gender in GENDERS]
    assets += [ROOT/'body_models/grail_g1_smplx'/name for name in ('g1_smplx_tpose.obj', 'g1_smplx_param.npz')]
    return {str(path): sha256(path) for path in assets}


def audit(args):
    """Oracle labels check the decoder only; labels never enter a model call."""
    ensure_bounded()
    torch.set_num_threads(2)
    paths = load_paths(args.paths)
    original = SmplhShape(paths['smplh_models']/'neutral/model.npz')
    bank = NativeShapeBank(original.rest_template[:1])
    native_manifest = ROOT/'experiments/offline_camera_retrain_v1/data/native30_official_memory_v3/manifest.jsonl'
    records = [r for r in map(json.loads, native_manifest.read_text().splitlines())
               if r['split']=='train' and r['dataset'] in ('egobody', 'trumans')]
    grail_manifest = ROOT/'runs/grail_ego_quality_v2/full/train/manifest.jsonl'
    records += [dict(json.loads((Path(r['path'])/'metadata.json').read_text()), path=r['path'])
                for r in list(map(json.loads, grail_manifest.read_text().splitlines()))[:args.per_group]]
    rows, counts = [], {}
    for row in records:
        group = row['group'] if row.get('dataset') != 'grail' else 'grail'
        # GRAIL may encode category in group; aggregate it as its own profile.
        if row['native_body'].get('template') == 'grail_g1_baked_v1':
            group = 'grail'
        if counts.get(group, 0) >= args.per_group:
            continue
        body = row['native_body']
        kind = 'grail_g1' if group=='grail' else body['model']
        gender = 0 if kind=='grail_g1' else GENDERS.index(body['gender'])
        beta = torch.tensor([body['betas']], dtype=torch.float32)
        expected = torch.as_tensor(np.load(Path(row['path'])/'rest.npy'))
        decoded = bank(kind, beta, torch.tensor([gender]))
        error = float((decoded['native_rest'][0]-expected).norm(dim=-1).max())
        # Native source labels use their own pelvis; the new root reference
        # must preserve absolute FK when only the translation channels shift.
        common = expected-expected[:1]+original.rest_template[:1]
        centered_error = float((decoded['rest'][0]-common).norm(dim=-1).max())
        mount = row.get('actual_virtual_mount')
        mount_error = float((decoded['head_camera_translation'][0]-torch.tensor(mount['translation_m'])).norm()) if mount else None
        if error > 2e-5 or centered_error > 2e-5:
            raise ValueError(f'Native shape decoder differs from source rest: {kind}, {error}')
        rows.append(dict(group=group, kind=kind, gender=body['gender'], sequence=row['sequence_id'],
            rest_max_error_m=error, common_reference_error_m=centered_error, virtual_mount_error_m=mount_error))
        counts[group] = counts.get(group, 0)+1
    if set(counts) != {'trumans', 'camera_wearer', 'interactee', 'grail'}:
        raise ValueError(f'Incomplete native profile audit: {counts}')
    result = dict(state='passed', completed_at_utc=now(), source=sha256(Path(__file__)),
        assets=source_assets(paths), rows=rows, counts=counts,
        max_rest_error_m=max(r['rest_max_error_m'] for r in rows),
        max_virtual_mount_error_m=max(r['virtual_mount_error_m'] or 0 for r in rows),
        training_started=False, inference_gt_shape_input=False,
        scope='Native decoder identity on oracle labels only; not a learned-model performance result.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key not in ('rows', 'assets')}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path, default=ROOT/'configs/paths.local.yaml')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/scene_ego128/preflight/native_shape.json')
    parser.add_argument('--per-group', type=int, default=12)
    args = parser.parse_args()
    if args.per_group < 1:
        parser.error('Positive audit sample count required')
    audit(args)


if __name__ == '__main__':
    main()
