"""Optional observed-scene conditioning for the public motion pipeline."""
import torch


class SceneConditionerMixin:
    """Encode static and current-frame observations when the scene branch is on."""

    def scene_tokens(self, batch):
        vit = self.core.scene_embedding
        p = vit.to_patch_embedding(batch['occupancy'])
        b, n, _ = p.shape
        x = torch.cat((vit.cls_token.expand(b, -1, -1), p), 1)
        x = vit.transformer(vit.dropout(x + vit.pos_embedding[:, :n + 1]))
        valid = (batch['occupancy'].reshape(b, 24, 4, 12, 4, 12)
                 .amax((1, 3, 5)) > 0).flatten(1)
        summary = vit.mlp_head(vit.to_latent(x[:, 0])) * valid.any(-1)[:, None]
        tokens = (self.patch_projection(x[:, 1:]) +
                  self.patch_position(self.patch_centres)[None]) * valid[..., None]
        dynamic = []
        if 'frame_scene_points' in batch:
            points, owners = batch['frame_scene_points'], batch['frame_scene_owner']
            mask = ((owners >= 0) & (owners < 100) &
                    torch.isfinite(points).all(-1))
            safe = torch.where(mask[..., None], points, 0.)
            feature = torch.cat((safe / 6.4, safe.new_zeros(*safe.shape[:-1], 8),
                                 self.owner_embedding(owners.clamp(0, 127))), -1)
            encoded = self.dynamic_embedding(feature) * mask[..., None]
            dynamic = [(encoded[i], mask[i]) for i in range(b)]
        else:
            dynamic = [(tokens.new_zeros(128, 0, 64),
                        valid.new_zeros(128, 0)) for _ in range(b)]
        return summary, tokens, valid, dynamic
