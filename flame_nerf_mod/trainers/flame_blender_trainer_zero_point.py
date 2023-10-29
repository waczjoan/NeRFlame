import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import trange
import numpy as np
import os
import imageio
from tqdm import tqdm
from nerf_pytorch.run_nerf_helpers import (
    img2mse,
    mse2psnr,
    to8b
)
from nerf_pytorch.nerf_utils import (
    render_path
)

from nerf_pytorch.nerf_utils import render
from nerf_pytorch.utils import load_obj_from_config

from flame_nerf_mod.trainers import BlenderTrainer
from FLAME import FLAME
from flame_nerf_mod.mesh_utils import (
    intersection_points_on_mesh,
    sample_extra_points_on_mesh,
    transform_points_to_single_number_representation,

)


class FlameBlenderTrainerExtraPoints(BlenderTrainer):
    """Trainer for Flame blender data."""
    def __init__(
            self,
            config,
            **kwargs
    ):
        super().__init__(
            **kwargs
        )
        flame_config = load_obj_from_config(cfg=config)
        self.model_flame = FLAME(flame_config).to(self.device)

        self.f_shape = nn.Parameter(torch.rand(1, 100).float().to(self.device))
        self.f_exp = nn.Parameter(torch.rand(1, 50).float().to(self.device))
        self.f_pose = nn.Parameter(torch.rand(1, 6).float().to(self.device))
        self.f_neck_pose = nn.Parameter(torch.rand(1, 3).float().to(self.device))
        self.f_trans = nn.Parameter(torch.zeros(1, 3).float().to(self.device))
        self.vertices_mal = nn.Parameter(8 * torch.ones(1, 1).float().to(self.device))

        f_lr = 0.001
        f_wd = 0.0001
        self.f_opt = torch.optim.Adam(
            params=[self.f_shape, self.f_exp, self.f_pose, self.f_neck_pose, self.f_trans, self.vertices_mal],
            lr=f_lr,
            weight_decay=f_wd
        )

        self.faces = self.flame_faces()
        self.eps = None

    def flame_vertices(self):
        vertices, _ = self.model_flame(
            self.f_shape, self.f_exp, self.f_pose,
            neck_pose=self.f_neck_pose, transl=self.f_trans
        )
        vertices = torch.squeeze(vertices)
        # vertices = vertices.cuda()

        vertices = vertices[:, [0, 2, 1]]
        vertices[:, 1] = -vertices[:, 1]
        vertices *= self.vertices_mal

        if self.tensorboard_logging:
            self.log_on_tensorboard(
                self.global_step,
                {
                    'train': {
                        'vertices_mal': self.vertices_mal,
                    }
                }
            )

        return vertices

    def flame_faces(self):
        faces = self.model_flame.faces
        faces = torch.tensor(faces.astype(np.int32))
        faces = torch.squeeze(faces)
        faces = faces.cuda()
        return faces

    def train(self, N_iters=200000 + 1):
        hwf, poses, i_test, i_val, i_train, images, render_poses = self.load_data()

        if self.render_test:
            render_poses = np.array(poses[i_test])
            render_poses = torch.Tensor(render_poses).to(self.device)

        hwf = self.cast_intrinsics_to_right_types(hwf=hwf)
        self.create_log_dir_and_copy_the_config_file()
        optimizer, render_kwargs_train, render_kwargs_test = self.create_nerf_model()

        if self.render_only:
            self.render(self.render_test, images, i_test, render_poses, hwf, render_kwargs_test)
            return self.render_only

        images, poses, rays_rgb, i_batch = self.prepare_raybatch_tensor_if_batching_random_rays(
            poses, images, i_train
        )

        print('Begin')
        print('TRAIN views are', i_train)
        print('TEST views are', i_test)
        print('VAL views are', i_val)

        start = self.start + 1
        for i in trange(start, N_iters):
            rays_rgb, i_batch, batch_rays, target_s = self.sample_random_ray_batch(
                rays_rgb,
                i_batch,
                i_train,
                images,
                poses,
                i
            )

            trans, loss, psnr, psnr0 = self.core_optimization_loop(
                optimizer, render_kwargs_train,
                batch_rays, i, target_s,
            )

            if self.tensorboard_logging:
                self.log_on_tensorboard(
                    i,
                    {
                        'train': {
                            'loss': loss,
                            'psnr': psnr
                        }
                    }
                )

            self.update_learning_rate(optimizer)

            self.rest_is_logging(
                i,
                render_poses,
                hwf,
                poses,
                i_test,
                images,
                loss,
                psnr, render_kwargs_train, render_kwargs_test,
                optimizer
            )

            self.global_step += 1

        self.writer.close()

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

        if self.global_step > 10:
            ray_idxs_intersection_mash = kwargs["ray_idxs_intersection_mash"]
            distance = kwargs["distance"].abs()
            mask = distance < self.eps/2
            _mask = mask[ray_idxs_intersection_mash]
            _rays = raw[ray_idxs_intersection_mash, : , 3]
            raw[ray_idxs_intersection_mash, :, 3] = (1-_rays) * _mask.bool() + _rays

            #mask_no_inter = torch.ones(raw.shape[0], 1)
            #mask_no_inter[ray_idxs_intersection_mash] = 0
            #raw[mask_no_inter.bool(), :, 3] = 0

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

        if self.global_step > 10:
            N_samples = 10

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

        ray_idxs_intersection_mash = None
        distance = None

        if self.global_step > 10:

            vertices = self.flame_vertices()
            ray_idxs_intersection_mash, pts_mesh, pts_diff_sum = intersection_points_on_mesh(
                faces=self.faces,
                vertices=vertices,
                ray_origins=rays_o,
                ray_directions=rays_d,
            )

            self.eps = 1 - self.global_step * 0.001
            if self.eps < 0.04:
                self.eps = 0.04
            # self.eps = 0.02 + torch.sin(torch.pi*100 + torch.tensor(torch.pi/200 * self.global_step)).abs()
            if self.tensorboard_logging:
                self.log_on_tensorboard(
                    self.global_step,
                    {
                        'train': {
                            'eps': self.eps,
                        }
                    }
                )

            extra_pts, distance = sample_extra_points_on_mesh(
                points=pts_mesh.squeeze(dim=1),
                ray_directions=rays_d,
                n_points=N_samples,
                eps=self.eps,
            )

            pts[ray_idxs_intersection_mash] = extra_pts[ray_idxs_intersection_mash]

            z_vals = transform_points_to_single_number_representation(
                ray_directions=rays_d,
                ray_origin=rays_o,
                points=pts
            )

            z_vals, z_vals_idx = torch.sort(z_vals, -1)
            idxs = torch.arange(z_vals.shape[0]).reshape(-1, 1) * torch.ones_like(z_vals)
            distance = distance[idxs.long(), z_vals_idx]

            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
            pts[ray_idxs_intersection_mash] += self.f_trans

            if self.global_step - self.global_step // 1000 * 1000 == 0:
                torch.save(pts, f'{self.global_step}_pts.pt')
                torch.save(pts_mesh, f'{self.global_step}_pts_mesh.pt')
                torch.save(extra_pts, f'{self.global_step}_extra_pts.pt')

        # [N_rays, N_samples, n_chanels]
        raw = network_query_fn(pts, viewdirs, network_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = self.raw2outputs(
            raw=raw,
            z_vals=z_vals,
            rays_d=rays_d,
            ray_idxs_intersection_mash=ray_idxs_intersection_mash,
            distance=distance,
            raw_noise_std=raw_noise_std,
            white_bkgd=white_bkgd,
            pytest=pytest
        )

        return rgb_map, disp_map, acc_map, weights, depth_map, z_vals, weights, raw

    def core_optimization_loop(
        self,
        optimizer, render_kwargs_train,
        batch_rays, i, target_s,
    ):
        rgb, disp, acc, extras = render(
            self.H, self.W, self.K,
            chunk=self.chunk, rays=batch_rays,
            verbose=i < 10, retraw=True,
            **render_kwargs_train
        )

        optimizer.zero_grad()
        self.f_opt.zero_grad()

        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][..., -1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        psnr0 = None
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()
        self.f_opt.step()

        return trans, loss, psnr, psnr0

    def rest_is_logging(
        self, i, render_poses, hwf, poses, i_test, images,
        loss, psnr, render_kwargs_train,
        render_kwargs_test, optimizer
    ):
        if i % self.i_weights == 0:
            path = os.path.join(self.basedir, self.expname, '{:06d}.tar'.format(i))
            data = {
                'global_step': self.global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if render_kwargs_train['network_fine'] is not None:
                data['network_fine_state_dict'] = render_kwargs_train['network_fine'].state_dict()
            torch.save(data, path)
            print('Saved checkpoints at', path)

        if i % self.i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, self.K, self.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(self.basedir, self.expname, '{}_spiral_{:06d}_'.format(self.expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        if i % self.i_testset == 0 and i > 0:
            testsavedir = os.path.join(self.basedir, self.expname, 'testset_f_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                vertice_out = self.flame_vertices()
                torch.save(vertice_out, f'test_{self.global_step}_vertice_out.pt')
                torch.save(self.faces, f'test_{self.global_step}_faces.pt')

                outmesh_path = os.path.join(testsavedir, 'face.obj')
                write_simple_obj(mesh_v=vertice_out.detach().cpu().numpy(), mesh_f=self.faces,
                                 filepath=outmesh_path)

                target_s = images[i_test]
                rgbs, _ = render_path(torch.Tensor(poses[i_test]).to(self.device), hwf, self.K, self.chunk,
                                      render_kwargs_test,
                                      gt_imgs=target_s, savedir=testsavedir)

                img_loss = img2mse(torch.Tensor(rgbs), torch.Tensor(target_s))
                loss = img_loss
                psnr = mse2psnr(img_loss)

                self.log_on_tensorboard(
                    i,
                    {
                        'test': {
                            'loss': loss,
                            'psnr': psnr
                        }
                    }
                )
            print('Saved test set')

        if i % self.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")


def write_simple_obj(mesh_v, mesh_f, filepath, verbose=False):
    with open(filepath, 'w') as fp:
        for v in mesh_v:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in mesh_f + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    if verbose:
        print('mesh saved to: ', filepath)