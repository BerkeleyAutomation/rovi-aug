import tensorflow as tf
import tensorflow_datasets as tfds
import sys
import torch
from omegaconf import DictConfig

from rovi_aug.mods.base_mod import BaseMod
from rlds_dataset_mod.mod_functions import add_obs_key

class R2RMod(BaseMod):
    r2r_augmentor = None
    batch_size = 200
    device = "cuda:0"
    masked_images_input_key = ""
    image_output_key = ""

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
    def load(cfg: DictConfig):
        """
        Uses information from the config file to load the mod.
        """
        R2RMod.device = cfg.device
        R2RMod.batch_size = cfg.robot_to_robot.batch_size
        R2RMod.masked_images_input_key = cfg.robot_to_robot.masked_images_input_key
        R2RMod.image_output_key = cfg.robot_to_robot.image_output_key
        
        controlnet_module_path = cfg.robot_to_robot.controlnet_module_path
        controlnet_checkpoint_folder_path = cfg.robot_to_robot.controlnet_checkpoint_folder_path
        target_robot = cfg.robot_to_robot.target_robot

        sys.path.append(controlnet_module_path)
        from refactorized_inference import ImageProcessor
        R2RMod.r2r_augmentor = ImageProcessor(
            base_model_path="runwayml/stable-diffusion-v1-5",
            controlnet_path=controlnet_checkpoint_folder_path,
            batch_size=R2RMod.batch_size,
            device=R2RMod.device,
            target_robot=target_robot,
        )