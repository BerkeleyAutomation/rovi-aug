import torch
from mirage2.view_augmentation.sampler.view_sampler import ViewSampler

class UniformViewSampler(ViewSampler):

    def __init__(self, device = "cuda:0", extrinsics_setting="lower") -> None:
        super().__init__()
        if extrinsics_setting == "lower":
            self.lower_translation_bound = torch.tensor([-0.1, -0.1, -0.1], device=device)
            self.upper_translation_bound = torch.tensor([0.1, 0.1, 0.1], device=device)
            
            # The rotation randomization is bounded by zyx euler angles in radians
            self.rotation_lower_bound = torch.tensor([-0.05, -0.05, -0.05], device=device)
            self.rotation_upper_bound = torch.tensor([0.05, 0.05, 0.05], device=device)
        elif extrinsics_setting == "upper":
            self.lower_translation_bound = torch.tensor([-0.25, -0.1, -0.25], device=device)
            self.upper_translation_bound = torch.tensor([0.25, 0.1, 0.25], device=device)
            
            # The rotation randomization is bounded by zyx euler angles in radians
            self.rotation_lower_bound = torch.tensor([-0.1, -0.1, -0.1], device=device)
            self.rotation_upper_bound = torch.tensor([0.1, 0.1, 0.1], device=device)
        elif extrinsics_setting == "ultra":
            self.lower_translation_bound = torch.tensor([-0.4, -0.1, -0.4], device=device)
            self.upper_translation_bound = torch.tensor([0.4, 0.1, 0.4], device=device)
            
            # The rotation randomization is bounded by zyx euler angles in radians
            self.rotation_lower_bound = torch.tensor([-0.1, -0.1, -0.1], device=device)
            self.rotation_upper_bound = torch.tensor([0.1, 0.1, 0.1], device=device)

        self.device = device

    def sample(self, start_view: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Samples SE(3) elements and returns as a batch tensor of size (B, ...)
        """
        new_view = start_view.clone()        
        translation_shifts = (self.upper_translation_bound - self.lower_translation_bound) * \
            torch.rand((new_view.shape[0], 3), device=new_view.device) + self.lower_translation_bound


        new_view[:, :3, -1] += translation_shifts

        random_rotations = self.get_random_rotation_matrix_batched(new_view.shape[0])
        new_view[:, :3, :3] = random_rotations @ new_view[:, :3, :3]

        return new_view

    def get_random_rotation_matrix_batched(self, count_rotation_matrices) -> torch.Tensor:
        tensor_1 = torch.ones(count_rotation_matrices, device=self.device)
        tensor_0 = torch.zeros(count_rotation_matrices, device=self.device)

        random_rotations = torch.rand((count_rotation_matrices, 3), device=self.device) * \
            (self.rotation_upper_bound - self.rotation_lower_bound) + self.rotation_lower_bound

        roll = random_rotations[:, 0]
        pitch = random_rotations[:, 1]
        yaw = random_rotations[:, 2]

        RX = torch.stack([
                torch.stack([tensor_1, tensor_0, tensor_0]),
                torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]),
                torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])]).permute(2, 0, 1)

        RY = torch.stack([
                        torch.stack([torch.cos(pitch), tensor_0, torch.sin(pitch)]),
                        torch.stack([tensor_0, tensor_1, tensor_0]),
                        torch.stack([-torch.sin(pitch), tensor_0, torch.cos(pitch)])]).permute(2, 0, 1)

        RZ = torch.stack([
                        torch.stack([torch.cos(yaw), -torch.sin(yaw), tensor_0]),
                        torch.stack([torch.sin(yaw), torch.cos(yaw), tensor_0]),
                        torch.stack([tensor_0, tensor_0, tensor_1])]).permute(2, 0, 1)

        R = RZ @ RY @ RX
        return R
