"""Body-shape prediction and kinematic preparation for the motion pipeline."""
import torch


class BodyGeometryMixin:
    """Keep body-format dispatch separate without changing checkpoint keys."""

    def predicted_body(self, observations, formats):
        features = self.shape_encoder.features(observations)
        betas = self.shape_encoder.readout(features)
        shape = self.shape(betas)
        shape.pop('head_cpf_translation')
        shape.update(predicted_native_betas=features.new_zeros(len(features), 10),
            template_logits=features.new_zeros(len(features), 3),
            predicted_template=torch.zeros(len(features), device=features.device, dtype=torch.long))
        native = formats > 0
        if native.any():
            decoded = self.native_body(features[native], formats[native])
            for key, values in decoded.items():
                shape[key] = shape[key].clone()
                shape[key][native] = values
        shape['predicted_betas'] = betas
        return shape
