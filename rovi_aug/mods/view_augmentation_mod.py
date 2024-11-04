import tensorflow as tf
import tensorflow_datasets as tfds
import math
import os
import sys
import torch
from omegaconf import DictConfig

from rovi_aug.mods.base_mod import BaseMod
from rovi_aug.view_augmentation.view_augmentation import ViewAugmentation
from rovi_aug.view_augmentation.sampler.view_sampler import ViewSampler

class ViewAugmentationMod(BaseMod):
    # Sadly, these variables have to be global given the abstractions from the tfds dataset
    batch_size = 80
    device = "cuda:0"
    image_input_key = ""
    image_output_key = ""

    trajectory_consistent_sampling = False

    # Internal non-configurable vars
    view_augmenter = None
    traj_idx = 0

    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        return features  # no feature changes

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        def augment_view(step):
            def process_images(trajectory_images):
                for i in range(0, math.ceil(len(trajectory_images)//ViewAugmentationMod.batch_size) + 1):
                    start = i*ViewAugmentationMod.batch_size
                    end = min((i+1)*ViewAugmentationMod.batch_size, len(trajectory_images))
                    camera_obs_batch = torch.from_numpy(trajectory_images[start:end]).float().to(ViewAugmentationMod.view_augmenter.device)
                    augmented_batch = ViewAugmentationMod.view_augmenter(camera_obs_batch, traj_id=ViewAugmentationMod.traj_idx, batch_id=i)
                    trajectory_images[start:end] = augmented_batch.cpu().numpy()
                ViewAugmentationMod.traj_idx += 1
                return trajectory_images

            step["observation"][ViewAugmentationMod.image_input_key] = tf.numpy_function(process_images, [step["observation"][ViewAugmentationMod.image_output_key]], tf.uint8)
            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].batch(ViewAugmentationMod.batch_size).map(augment_view).unbatch()
            return episode

        return ds.map(episode_map_fn)

    @classmethod
    def load(cfg: DictConfig):
        """
        Uses information from the config file to load the mod.
        """
        ViewAugmentationMod.device = cfg.device
        ViewAugmentationMod.batch_size = cfg.view_augmentation.batch_size

        ViewAugmentationMod.image_input_key = cfg.view_augmentation.image_input_key
        ViewAugmentationMod.image_output_key = cfg.view_augmentation.image_output_key

        view_sampler = ViewSampler.from_config(cfg)

        # Hack to import ZeroNVS, since packaging doesn't seem to work
        sys.path.append(os.path.expanduser(cfg.view_augmentation.zeronvs_module_path))
        # import threestudio.utils.misc as misc
        # misc.EXT_DEVICE = device

        from threestudio.models.guidance import zero123_guidance

        ViewAugmentationMod.view_augmenter = ViewAugmentation(
            view_sampler,
            sample_img_path=cfg.view_augmentation.sample_img_path,
            checkpoint_path=cfg.view_augmentation.zeronvs_checkpoint_path,
            zeronvs_config_path=cfg.view_augmentation.zeronvs_config_path,
            zero123_guidance_module=zero123_guidance,
            original_size=256,
            device=ViewAugmentationMod.device,
        )