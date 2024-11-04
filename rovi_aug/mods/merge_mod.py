import tensorflow as tf
import tensorflow_datasets as tfds
import sys
from omegaconf import DictConfig

from rovi_aug.mods.base_mod import BaseMod
from rlds_dataset_mod.mod_functions import add_obs_key

class AugMergeMod(BaseMod):
    aug_merger = None

    robot_aug_imgs_input_key = ""
    inpainted_background_input_key = ""
    merged_output_key = ""

    video_inpaint_module_path = ""

    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        img_size = features["steps"]["observation"][AugMergeMod.inpainted_background_input_key].shape[0]
        return add_obs_key(features, AugMergeMod.merged_output_key, tfds.features.Tensor(shape=(img_size, img_size, 3), dtype=tf.uint8))

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        if AugMergeMod.aug_merger is None:
            sys.path.append(AugMergeMod.video_inpaint_module_path)
            from merge_two_images_refactoriezed import ImageProcessor
            AugMergeMod.aug_merger = ImageProcessor('.')

        def augment_view(step):
            def process_images(background_images, objects):
                augmented_img = AugMergeMod.aug_merger.paste_objects(background_images, objects)
                return augmented_img
            
            step["observation"][AugMergeMod.merged_output_key] = tf.numpy_function(process_images, [step["observation"][AugMergeMod.inpainted_background_input_key], step["observation"][AugMergeMod.robot_aug_imgs_input_key]], tf.uint8)
            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].map(augment_view)
            return episode

        return ds.map(episode_map_fn)

    @classmethod
    def load(cfg: DictConfig):
        """
        Uses information from the config file to load the mod.
        """
        AugMergeMod.robot_aug_imgs_input_key = cfg.aug_merge.robot_aug_imgs_input_key
        AugMergeMod.inpainted_background_input_key = cfg.aug_merge.inpainted_background_input_key
        AugMergeMod.merged_output_key = cfg.aug_merge.merged_output_key
        AugMergeMod.video_inpaint_module_path = cfg.aug_merge.video_inpaint_module_path