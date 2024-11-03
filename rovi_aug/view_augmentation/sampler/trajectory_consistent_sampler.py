import torch
from mirage2.view_augmentation.sampler.view_sampler import ViewSampler

class TrajectoryConsistentSampler(ViewSampler):

    def __init__(self, base_sampler: ViewSampler):
        """
        Initializes a trajectory consistent sampler
        :param base_sampler: A ViewSampler object that can sample views
        """
        super().__init__()
        self.base_sampler = base_sampler
        self.trajectory_sample_cache = {}

    def sample(self, start_view: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Samples SE(3) elements and returns as a batch tensor of size (B, ...)
        For the same trajectory id and the same start view, the same output view is returned
        """
        assert "traj_id" in kwargs
        traj_id = kwargs["traj_id"]
        if traj_id not in self.trajectory_sample_cache:
            self.trajectory_sample_cache[traj_id] = self.base_sampler.sample(start_view[0:1], **kwargs)
        return self.trajectory_sample_cache[traj_id].expand(start_view.shape[0], 4, 4)