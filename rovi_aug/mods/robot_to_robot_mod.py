import tensorflow as tf
import tensorflow_datasets as tfds
import sys
import torch
import numpy as np
from omegaconf import DictConfig

from rovi_aug.mods.base_mod import BaseMod, add_obs_key

class R2RMod(BaseMod):
    batch_size = 200
    device = "cuda:0"
    masked_images_input_key = ""
    image_output_key = ""

    controlnet = None
    pipe = None
    to_tensor_transform = None
    to_pil_transform = None
    prompt = ""

    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        img_size = features["steps"]["observation"][R2RMod.masked_images_input_key].shape[0]
        return add_obs_key(features, "robot_aug_imgs", tfds.features.Tensor(shape=(img_size, img_size, 3), dtype=tf.uint8))

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:

        def augment_view(step):
            def process_images(trajectory_images):
                with torch.no_grad():
                    return R2RMod.r2r_augmentor.process_folders(trajectory_images)

            step["observation"][R2RMod.image_output_key] = tf.numpy_function(process_images, [step["observation"][R2RMod.masked_images_input_key]], tf.uint8)
            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].batch(R2RMod.batch_size).map(augment_view).unbatch()
            return episode

        return ds.map(episode_map_fn)

    @classmethod
    def load(cls, cfg: DictConfig):
        """
        Uses information from the config file to load the mod.
        """
        from torchvision import transforms
        from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler

        R2RMod.device = cfg.device
        R2RMod.batch_size = cfg.robot_to_robot.batch_size
        R2RMod.masked_images_input_key = cfg.robot_to_robot.masked_images_input_key
        R2RMod.image_output_key = cfg.robot_to_robot.image_output_key
        
        controlnet_checkpoint_folder_path = cfg.robot_to_robot.controlnet_checkpoint_folder_path
        target_robot = cfg.robot_to_robot.target_robot

        R2RMod.controlnet = ControlNetModel \
            .from_pretrained(controlnet_checkpoint_folder_path, torch_dtype=torch.float16) \
            .to(device=R2RMod.device)
        R2RMod.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=R2RMod.controlnet, torch_dtype=torch.float16
        ).to(device=R2RMod.device)
        R2RMod.pipe.scheduler = UniPCMultistepScheduler.from_config(R2RMod.pipe.scheduler.config)
        R2RMod.pipe.safety_checker = None
        R2RMod.pipe.requires_safety_checker = False
        R2RMod.pipe.enable_sequential_cpu_offload()
        R2RMod.generator = torch.manual_seed(1)

        R2RMod.target_robot = target_robot
        R2RMod.to_tensor_transform = transforms.ToTensor()
        R2RMod.to_pil_transform = transforms.ToPILImage()

        R2RMod.prompt = f"create a high quality image with a {target_robot} robot and white background"

    @staticmethod
    def process_image_trajectory(image_trajectory): # image_trajectory is M x H x W x 3
        """
        Args:
            image_trajectory: np.ndarray of shape (M, H, W, 3), the input image trajectory
        """
        list_of_input_images = [R2RMod.to_pil_transform(image_trajectory[i]) for i in range(image_trajectory.shape[0])]
        batch_images = list_of_input_images
        prompts = [R2RMod.prompt] * len(batch_images)
        generated_images = R2RMod.pipe(
                prompt=prompts, 
                num_inference_steps=50, 
                generator=R2RMod.generator, 
                image=batch_images, 
                control_image=batch_images
        ).images
        generated_images = [np.array(R2RMod.to_pil(R2RMod.to_tensor(generated_image).clamp(0, 1))) for generated_image in generated_images]
        return np.stack(generated_images)