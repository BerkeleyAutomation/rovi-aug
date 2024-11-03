from typing import Any
from ldm.models.diffusion import options
from omegaconf import OmegaConf
import numpy as np
import torch
from torchvision.transforms import Resize
from mirage2.view_augmentation.sampler.view_sampler import ViewSampler

options.LDM_DISTILLATION_ONLY = True

class ViewAugmentation:

    def __init__(self, 
                 view_sampler: ViewSampler,
                 sample_img_path: str,
                 checkpoint_path: str,
                 zeronvs_config_path: str,
                 zero123_guidance_module: Any,
                 device: str = "cuda:0",
                 original_size: int = 128,
                 camera_fov_deg: float = 45.0):
        """
        Creates a view augmentation, which produces randomly sampled views from an input batch
        :param view_sampler: A ViewSampler object that can sample views
        :param device: The device to run the augmentation on
        :param camera_fov_deg: The field of view of the camera
        """

        self.view_sampler = view_sampler

        # The image just has to exist but otherwise it doesn't really matter
        image_path = sample_img_path
        self.guidance_cfg = dict(
            pretrained_model_name_or_path= checkpoint_path,
            pretrained_config= zeronvs_config_path,
            guidance_scale= 9.5,
            cond_image_path =image_path,
            min_step_percent=[0,.75,.02,1000],
            max_step_percent=[1000, 0.98, 0.025, 2500],
            vram_O=False
        )
        
        self.guidance = zero123_guidance_module.Zero123Guidance(OmegaConf.create(self.guidance_cfg))
        self.guidance.device = device

        # The precomputed scale here is how close the camera is to the scene.
        self.guidance.cfg.precomputed_scale=.9

        self.device = device
        self.fov_tensor = torch.from_numpy(np.array([camera_fov_deg])).to(self.device).to(torch.float32)
        self.resizer = Resize((256, 256), antialias=True)
        self.original_sizer = Resize((original_size, original_size), antialias=True)

    def __call__(self, view_images, **kwargs):
        """
        :param view_images: A tensor containing all of the input views, batched
        """
        needs_to_be_reverted = False
        if view_images.numel() == 0:
            return view_images

        if view_images.max() > 1:
            needs_to_be_reverted = True

        if needs_to_be_reverted:
            view_images = view_images / 255.0
            view_images = view_images.permute(0, 3, 1, 2)
            view_images = self.resizer(view_images)

        c_crossattn, c_concat = self.guidance.get_img_embeds(
            view_images
        )

        cond_camera = torch.eye(4, device=self.device) \
                            .unsqueeze(0) \
                            .expand(view_images.shape[0], 4, 4)
        target_views = self.view_sampler.sample(cond_camera, **kwargs)

        camera_batch = {
            "target_cam2world": target_views,
            "cond_cam2world": cond_camera,
            "fov_deg": self.fov_tensor.repeat(view_images.shape[0])
        }

        cond = self.guidance.get_cond_from_known_camera(
            camera_batch,
            c_crossattn=c_crossattn,
            c_concat=c_concat,
        )
        del c_crossattn
        del c_concat
        del camera_batch
        torch.cuda.empty_cache()

        novel_views = self.guidance.gen_from_cond(cond, post_process=False)
        if needs_to_be_reverted:
            novel_views = self.original_sizer(novel_views)
            novel_views = novel_views.permute(0, 2, 3, 1)
            novel_views = torch.clamp(novel_views * 255.0, 0.0, 255.0)
        return novel_views