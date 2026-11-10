"""Adapters for the pinned public EgoAllo and EgoGVAE inference implementations.

Only camera observations, a public sensor profile and fixed train/neutral
calibration enter these adapters. Targets and personal body coefficients do not.
"""
import contextlib
import dataclasses
import importlib
import io
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def upstream_imports(repo):
    """Isolate upstream's generic model/utils namespaces from SceneMI."""
    saved = {k: v for k, v in sys.modules.items()
             if k.split('.')[0] in ('model', 'utils')}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, str(repo))
    try:
        yield
    finally:
        sys.path.remove(str(repo))
        for k in list(sys.modules):
            if k.split('.')[0] in ('model', 'utils'):
                del sys.modules[k]
        sys.modules.update(saved)


class PublicBaselines:
    def __init__(self, config, device, mount):
        from egoallo import fncsmpl, fncsmpl_extensions
        self.c, self.device = config, torch.device(device)
        self.body = fncsmpl.SmplhModel.load(Path(config['smplh_model'])).to(device)
        self.z_to_y = torch.tensor([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]], device=device)
        neutral = self.body.with_shape(torch.zeros(1, 16, device=device))
        self.head_cpf = fncsmpl_extensions.get_T_head_cpf(neutral)[0, 4:]
        head_camera = neutral.verts_zero[0, [2802, 6262]].mean(0) - neutral.joints_zero[0, 14]
        head_camera = head_camera + head_camera.new_tensor([0., 0., .02])
        self.virtual_camera_cpf = self.head_cpf - head_camera
        self.pv_rotation = torch.tensor(mount['rotation'], device=device, dtype=torch.float32)
        self.pv_translation = torch.tensor(mount['translation_m'], device=device, dtype=torch.float32)
        self.egoallo = self.egogvae = None
        self._guidance = None

    def cpf(self, camera, sensor_profile):
        from egoallo.transforms import SO3
        camera = torch.as_tensor(camera, device=self.device, dtype=torch.float32)
        six = camera[..., 3:]
        first = torch.nn.functional.normalize(six[..., :3], dim=-1)
        second = six[..., 3:] - (first * six[..., 3:]).sum(-1, keepdim=True) * first
        second = torch.nn.functional.normalize(second, dim=-1)
        rotation = torch.stack((first, second, torch.cross(first, second, dim=-1)), -1)
        pv = torch.as_tensor(sensor_profile, device=self.device) == 1
        head_rotation = rotation.clone()
        head_rotation[pv] = rotation[pv] @ self.pv_rotation.T
        offset = self.virtual_camera_cpf.expand(len(camera), 3).clone()
        offset[pv] = self.head_cpf - self.pv_translation
        position = camera[..., :3] * 2 + (head_rotation @ offset[:, None, :, None])[..., 0]
        z_position = position @ self.z_to_y
        z_rotation = self.z_to_y.T @ head_rotation
        poses = torch.cat((SO3.from_matrix(z_rotation).wxyz, z_position), -1)
        # Same 128 observed frames for every model: no extra GT-derived frame.
        return torch.cat((poses[:, :1], poses), 1)

    def load(self, name):
        if name == 'egoallo' and self.egoallo is None:
            from egoallo.inference_utils import load_denoiser
            self.egoallo = load_denoiser(ROOT / self.c['egoallo_checkpoint']).eval().to(self.device)
        if name == 'egogvae' and self.egogvae is None:
            with upstream_imports((ROOT / self.c['egogvae_repository']).resolve()):
                network = importlib.import_module('model.network')
                self.gvae_rotation = importlib.import_module('model.utils_rotation')
                self.egogvae = network.Head2Motion(head_dim=16, motion_dim=135, out_dim=163)
            state = torch.load(ROOT / self.c['egogvae_checkpoint'], map_location='cpu', weights_only=True)
            self.egogvae.load_state_dict(state['model_state_dict'], strict=True)
            self.egogvae.requires_grad_(False).eval().to(self.device)

    def gvae_head(self, cpf):
        from egoallo.transforms import SE3, SO3
        relative = (SE3(cpf[:, :-1]).inverse() @ SE3(cpf[:, 1:])).as_matrix()
        rotation = SO3(cpf[:, 1:, :4])
        forward = rotation @ cpf.new_tensor([0., 0., 1.])
        canonical = SO3.from_z_radians(-torch.atan2(forward[..., 1], forward[..., 0]))
        r6 = self.gvae_rotation.matrix_to_rotation_6d
        return torch.cat((r6(relative[..., :3, :3]), relative[..., :3, 3],
                          cpf[:, 1:, 6:7], r6((canonical @ rotation).as_matrix())), -1)

    @torch.inference_mode()
    def gvae_sample(self, cpf, case_ids):
        from egoallo.transforms import SO3
        self.load('egogvae')
        head = self.gvae_head(cpf)
        outputs = []
        # Keep each sample's RNG independent of batching and restart position.
        for i, case_id in enumerate(case_ids):
            torch.manual_seed(self.c['seed'] + int(case_id))
            with contextlib.redirect_stdout(io.StringIO()):
                outputs.append(self.egogvae.forward_test(head[i:i + 1]))
        theta = torch.cat([x['head_theta'] for x in outputs])
        betas = torch.cat([x['head_betas'] for x in outputs]).mean(1, keepdim=True).expand(-1, 128, -1)
        body = self.gvae_rotation.rotation_6d_to_matrix(theta.reshape(len(head), 128, 21, 6))
        hands = torch.eye(3, device=self.device).expand(len(head), 128, 30, 3, 3)
        return self.world_joints(betas, SO3.from_matrix(torch.cat((body, hands), 2)).wxyz, cpf[:, 1:])

    def world_joints(self, betas, local_quats, cpf):
        from egoallo.fncsmpl_extensions import get_T_world_root_from_cpf_pose
        from egoallo.transforms import SE3
        posed = self.body.with_shape(betas).with_pose(
            SE3.identity(self.device, torch.float32).wxyz_xyz, local_quats)
        root = get_T_world_root_from_cpf_pose(posed, cpf)
        posed = posed.with_new_T_world_root(root)
        z = torch.cat((root[..., None, 4:], posed.Ts_world_joint[..., :21, 4:]), -2)
        return z @ self.z_to_y.T

    @torch.inference_mode()
    def allo_sample(self, cpf, case_ids):
        """Vectorize the released 30-step sampler over independent conditions.

        Both overlapping passes (0:128 and 96:128), projection, denominator,
        noise schedule and stochastic sigma are retained exactly.
        """
        from egoallo.network import EgoDenoiseTraj
        from egoallo.sampling import CosineNoiseScheduleConstants, quadratic_ts
        from egoallo.transforms import SE3
        self.load('egoallo')
        b, frames = len(cpf), cpf.shape[1] - 1
        generators = [torch.Generator(device=self.device).manual_seed(self.c['seed'] + int(i)) for i in case_ids]
        def noise():
            return torch.stack([torch.randn((frames, self.egoallo.get_d_state()),
                               device=self.device, generator=g) for g in generators])
        x = noise()
        schedule = CosineNoiseScheduleConstants.compute(1000).to(device=self.device)
        ab, alpha = schedule.alpha_bar_t, schedule.alpha_t
        sigma = torch.cat((torch.zeros(1, device=self.device),
            torch.sqrt((1 - ab[:-1]) / (1 - ab[1:]) * (1 - alpha)) * .8))
        delta = (SE3(cpf[:, :-1]).inverse() @ SE3(cpf[:, 1:])).wxyz_xyz
        weights = torch.from_numpy(np.minimum(32, np.minimum(np.arange(1, frames + 1),
                        np.arange(1, frames + 1)[::-1])) / 32).to(self.device, torch.float32)
        ts = quadratic_ts()
        for t, next_t in zip(ts[:-1], ts[1:]):
            output = torch.zeros_like(x); weight_sum = torch.zeros(1, frames, 1, device=self.device)
            for start in range(0, frames, 96):
                end = min(start + 128, frames)
                w = weights[None, :end - start, None]
                output[:, start:end] += self.egoallo(
                    x[:, start:end], torch.full((b,), int(t), device=self.device),
                    T_cpf_tm1_cpf_t=delta[:, start:end], T_world_cpf=cpf[:, start + 1:end + 1],
                    project_output_rotmats=False, hand_positions_wrt_cpf=None, mask=None) * w
                weight_sum[:, start:end] += w
            output = EgoDenoiseTraj.unpack(output / weight_sum, include_hands=True, project_rotmats=True).pack()
            x = (torch.sqrt(ab[next_t]) * output + torch.sqrt(1 - ab[next_t] - sigma[t] ** 2)
                 * (x - torch.sqrt(ab[t]) * output) / torch.sqrt(1 - ab[t] + 1e-1)
                 + sigma[t] * noise())
        return EgoDenoiseTraj.unpack(x, include_hands=True)

    def guided(self, traj, cpf):
        import jax
        import jax.numpy as jnp
        from egoallo import fncsmpl_jax, guidance_optimizer_jax as opt
        from egoallo.transforms import SO3
        if self._guidance is None:
            body = fncsmpl_jax.SmplhModel(**{
                name: jnp.asarray(getattr(self.body, name).detach().cpu().numpy())
                for name in ('faces', 'J_regressor', 'weights', 'posedirs', 'v_template', 'shapedirs')},
                parent_indices=jnp.asarray(self.body.parent_indices))
            params = opt.JaxGuidanceParams.defaults('no_hands', 'post')
            def one(ts, betas, rotations, hands, contacts):
                return opt._optimize(Ts_world_cpf=ts, body=body, betas=betas,
                    body_rotmats=rotations, hand_rotmats=hands, contacts=contacts,
                    guidance_params=params, hamer_detections=None, aria_detections=None, verbose=False)[0]
            self._guidance = jax.jit(jax.vmap(one))
        arrays = [jnp.asarray(x.detach().cpu().numpy()) for x in
                  (cpf[:, 1:], traj.betas, traj.body_rotmats, traj.hand_rotmats, traj.contacts)]
        quats = torch.as_tensor(np.asarray(self._guidance(*arrays)).copy(), device=self.device)
        rotations = SO3(quats).as_matrix()
        return dataclasses.replace(traj, body_rotmats=rotations[:, :, :21], hand_rotmats=rotations[:, :, 21:])

    def allo_joints(self, traj, cpf):
        from egoallo.transforms import SO3
        return self.world_joints(traj.betas, SO3.from_matrix(torch.cat(
            (traj.body_rotmats, traj.hand_rotmats), 2)).wxyz, cpf[:, 1:])
