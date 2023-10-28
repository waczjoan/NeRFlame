from flame_nerf_mod.trainers import BlenderTrainer


class FlameBlenderTrainer(BlenderTrainer):
    """Trainer for Flame blender data."""
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(
            **kwargs
        )
