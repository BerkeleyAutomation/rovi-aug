import tensorflow as tf
import tensorflow_datasets as tfds
import sys
import torch
import numpy as np
from omegaconf import DictConfig

from rovi_aug.mods.base_mod import BaseMod
from rlds_dataset_mod.mod_functions import add_obs_key

class VideoInpaintMod(BaseMod):
    video_inpainter = None

    batch_size = 400
    device = "cuda:0"
    image_input_key = ""
    mask_input_key = ""
    image_output_key = ""

    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        img_size = features["steps"]["observation"][VideoInpaintMod.image_input_key].shape[0]
        return add_obs_key(features, VideoInpaintMod.image_output_key, tfds.features.Tensor(shape=(img_size, img_size, 3), dtype=tf.uint8))

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        def augment_view(step):
            def process_images(trajectory_images, masked_images):
                try:
                    non_stacked_imgs = VideoInpaintMod.video_inpainter.main_worker(trajectory_images, masked_images)
                    return non_stacked_imgs
                except Exception as e:
                    print(e)
                    return trajectory_images
                # return np.stack(non_stacked_imgs, axis=0).astype(np.uint8)

            step["observation"][VideoInpaintMod.image_output_key] = tf.numpy_function(process_images, [step["observation"][VideoInpaintMod.image_input_key], step["observation"][VideoInpaintMod.mask_input_key]], tf.uint8)
            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].batch(VideoInpaintMod.batch_size).map(augment_view).unbatch()
            return episode

        return ds.map(episode_map_fn)

    @classmethod
    def load(cfg: DictConfig):
        """
        Uses information from the config file to load the mod.
        """
        VideoInpaintMod.device = cfg.device
        VideoInpaintMod.batch_size = cfg.video_inpaint.batch_size
        VideoInpaintMod.image_input_key = cfg.video_inpaint.image_input_key
        VideoInpaintMod.image_output_key = cfg.video_inpaint.image_output_key
        
        video_inpaint_checkpoint_path = cfg.video_inpaint.video_inpaint_checkpoint_path
        video_inpaint_module_path = cfg.video_inpaint.video_inpaint_module_path

        sys.path.append(video_inpaint_module_path)
        from refractorized_inference import VideoProcessor
        args = {
            "model": "e2fgvi_hq",
            "width": 256,
            "height": 256,
            "step": 10,
            "num_ref": -1,
            "neighbor_stride": 5,
            "savefps": 24,
            "set_size": True,
            "save_frame": "./",
            "video.split": "test.mp4",
            "use_mp4": True,
            "ckpt": video_inpaint_checkpoint_path
        }
        VideoInpaintMod.video_inpainter = VideoProcessor(args)