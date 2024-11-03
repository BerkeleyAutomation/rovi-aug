import tensorflow as tf
import tensorflow_datasets as tfds
import math
import os
import sys
import torch
from omegaconf import DictConfig

from rovi_aug.mods.base_mod import BaseMod
from rovi_aug.view_augmentation.view_augmentation import ViewAugmentation
from rovi_aug.view_augmentation.sampler.uniform_view_sampler import UniformViewSampler

class ViewAugmentationMod(BaseMod):
    # Sadly, these variables have to be global given the abstractions from the tfds dataset
    batch_size = 80
    device = "cuda:0"
    zeronvs_path = ""
    zeronvs_checkpoint_path = ""
    zeronvs_config_path = ""
    sample_img_path = ""

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
        if ViewAugmentationMod.view_augmenter is None:
            view_sampler = UniformViewSampler(device=ViewAugmentationMod.device)

            # Hack to import ZeroNVS, since packaging doesn't seem to work
            sys.path.append(os.path.expanduser(ViewAugmentationMod.zeronvs_path))
            # import threestudio.utils.misc as misc
            # misc.EXT_DEVICE = device

            from threestudio.models.guidance import zero123_guidance

            ViewAugmentationMod.view_augmenter = ViewAugmentation(
                view_sampler,
                sample_img_path=ViewAugmentationMod.sample_img_path,
                checkpoint_path=ViewAugmentationMod.zeronvs_checkpoint_path,
                zeronvs_config_path=ViewAugmentationMod.zeronvs_config_path,
                zero123_guidance_module=zero123_guidance,
                original_size=256,
                device=ViewAugmentationMod.device,
            )

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

            step["observation"]["robot_aug_imgs"] = tf.numpy_function(process_images, [step["observation"]["merged_robot_aug"]], tf.uint8)
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
        ViewAugmentationMod.zeronvs_path = cfg.view_augmentation.zeronvs_path
        ViewAugmentationMod.batch_size = cfg.view_augmentation.batch_size
        ViewAugmentationMod.zeronvs_checkpoint_path = cfg.view_augmentation.zeronvs_checkpoint_path
        ViewAugmentationMod.zeronvs_config_path = cfg.view_augmentation.zeronvs_config_path
        ViewAugmentationMod.sample_img_path = cfg.view_augmentation.sample_img_path