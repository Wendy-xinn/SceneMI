"""GRAIL body assets in a namespace separate from the sealed native datasets.

The released G1-proportioned SMPL-X template changes rest joints and skin
geometry. Encoding it as ordinary zero-beta SMPL-X would silently use the
wrong body in scene queries. Reuse the existing skin implementation with an
independent model and cache for each explicitly declared template.
"""
from __future__ import annotations

from .grail_source import body_model_factory, g1_template_assets

G1_TEMPLATE = 'grail_g1_baked_v1'
STANDARD_TEMPLATE = 'smplx_neutral_flat_v1'


class GrailSkinAnchorCache:
    def __init__(self, device='cpu', limit=64):
        self.device, self.limit = device, limit
        self.caches = {}

    def get(self, info):
        from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache

        template = info.get('template')
        if template not in (G1_TEMPLATE, STANDARD_TEMPLATE):
            raise ValueError('GRAIL requires an explicit verified body template')
        if info.get('model') != 'smplx' or info.get('gender') != 'neutral':
            raise ValueError('GRAIL body assets must be neutral SMPL-X')
        if template == G1_TEMPLATE:
            if abs(float(info['scale']) - 1.) > 1e-7:
                raise ValueError('G1 template already contains scale; do not apply source scale again')
            if any(abs(float(beta)) > 1e-7 for beta in info['betas']):
                raise ValueError('G1 template has baked shape and requires zero betas')
            assets = g1_template_assets()
            supplied = info.get('template_sha256')
            if supplied is not None and supplied != assets['sha256']:
                raise ValueError('GRAIL body template fingerprints changed')
        if template not in self.caches:
            cache = Native30SkinAnchorCache(device=self.device, limit=self.limit)
            cache.models[('smplx', 'neutral')] = body_model_factory(
                'g1' if template == G1_TEMPLATE else 'standard')
            self.caches[template] = cache
        return self.caches[template].get(info)


def validate_body_assets(info):
    """Build CPU skin assets and check their rest skeleton against the model."""
    import numpy as np
    from smplx.lbs import vertices2joints
    import torch

    cache = GrailSkinAnchorCache(device='cpu')
    asset = cache.get(info)
    model = cache.caches[info['template']].models[('smplx', 'neutral')]
    with torch.no_grad():
        expected = vertices2joints(model.J_regressor, model.v_template[None])[0].numpy()
    actual = asset['joints'].cpu().numpy()
    error = float(np.max(np.abs(expected - actual)))
    if error > 1e-6:
        raise ValueError('GRAIL rest skeleton and skin asset disagree')
    return dict(template=info['template'], rest_max_error_m=error,
                anchors=len(asset['vertices']), device='cpu',
                template_sha256=info.get('template_sha256'),
                native_body_assets_verified=True)
