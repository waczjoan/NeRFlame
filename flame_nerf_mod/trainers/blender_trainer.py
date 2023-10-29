"""Blender trainer module - trainer for blender data."""
import torch
from nerf_pytorch.trainers.Trainer import Trainer

from flame_nerf_mod.models import NeRF
from flame_nerf_mod.blender.load_blender import load_blender_data
import torch.nn.functional as F
import numpy as np


class BlenderTrainer(Trainer):
    """Trainer for blender data."""

    def __init__(
        self,
        half_res,
        white_bkgd,
        testskip=8,
        near=2.0,
        far=6.0,
        **kwargs
    ):
        """Initialize the blender trainer.

        In addition to original nerf_pytorch BlenderTrainer,
        the trainer contains the spheres used in the training process.
        """

        self.half_res = half_res
        self.testskip = testskip
        self.white_bkgd = white_bkgd

        self.near = near
        self.far = far

        super().__init__(
            **kwargs
        )

    def load_data(self):
        images, poses, render_poses, hwf, i_split = load_blender_data(
            self.datadir, self.half_res, self.testskip
        )
        i_train, i_val, i_test = i_split

        if self.white_bkgd:
            images = (images[..., :3] *
                      images[..., -1:] + (1. - images[..., -1:])
                )
        else:
            images = images[..., :3]

        render_poses = torch.Tensor(render_poses).to(self.device)
        return hwf, poses, i_test, i_val, i_train, images, render_poses

    def create_nerf_model(self):
        return self._create_nerf_model(model=NeRF)

    def sample_main_points(
        self,
        near,
        far,
        perturb,
        N_rays,
        N_samples,
        viewdirs,
        network_fn,
        network_query_fn,
        rays_o,
        rays_d,
        raw_noise_std,
        white_bkgd,
        pytest,
        lindisp,
        **kwargs
    ):

        t_vals = torch.linspace(0., 1., steps=N_samples)
        if not lindisp:
            z_vals = near * (1. - t_vals) + far * t_vals
        else:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)

        z_vals = z_vals.expand([N_rays, N_samples])

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)

            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.Tensor(t_rand)

            z_vals = lower + (upper - lower) * t_rand

        # [N_rays, N_samples, 3]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        if self.global_step - self.global_step//1000 * 1000 == 0:
            torch.save(pts, f'{self.global_step}_pts.pt')


        # [N_rays, N_samples, n_chanels]
        raw = network_query_fn(pts, viewdirs, network_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = self.raw2outputs(
            raw=raw,
            z_vals=z_vals,
            rays_d=rays_d,
            raw_noise_std=raw_noise_std,
            white_bkgd=white_bkgd,
            pytest=pytest
        )

        return rgb_map, disp_map, acc_map, weights, depth_map, z_vals, weights, raw

    def raw2outputs(
        self,
        raw,
        z_vals,
        rays_d,
        raw_noise_std=0,
        white_bkgd=False,
        pytest=False,
        **kwargs
    ):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
            ray_idxs_intersection_mash
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat(
            [dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)],
            -1
        )  # [N_rays, N_samples]

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[..., 3].shape) * raw_noise_std

            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
                noise = torch.Tensor(noise)

        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

        if self.global_step - self.global_step//1000 * 1000 == 0:
            torch.save(alpha, f'{self.global_step}_alpha.pt')

        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:,
                          :-1]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1. / torch.max(
            1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)
        )
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, depth_map
