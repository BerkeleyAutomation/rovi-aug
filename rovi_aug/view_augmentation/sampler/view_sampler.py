import torch
from omegaconf import DictConfig

class ViewSampler:
    def sample(self, start_view: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Samples SE(3) elements and returns as a batch tensor of size (B, ...)
        """
        raise NotImplementedError
    
    @staticmethod
    def from_config(cfg: DictConfig) -> "ViewSampler":
        """
        Initializes the sampler from the overall config
        """

        # Import here to avoid circular dependencies
        from rovi_aug.view_augmentation.sampler.uniform_view_sampler import UniformViewSampler
        from rovi_aug.view_augmentation.sampler.trajectory_consistent_sampler import TrajectoryConsistentSampler

        view_sampler = UniformViewSampler(
            device=cfg.device,
            extrinsics_setting=cfg.view_augmentation.sampler.extrinsics_setting,
        )

        if cfg.view_augmentation.sampler.trajectory_consistent_sampling:
            view_sampler = TrajectoryConsistentSampler(view_sampler)

        return view_sampler
        