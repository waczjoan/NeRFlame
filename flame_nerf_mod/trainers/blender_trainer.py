"""Blender trainer module - trainer for blender data."""
import torch
from nerf_pytorch.trainers.Trainer import Trainer

from flame_nerf_mod.models import NeRF
from flame_nerf_mod.blender.load_blender import load_blender_data


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
