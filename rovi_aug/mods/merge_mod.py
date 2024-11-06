import tensorflow as tf
import tensorflow_datasets as tfds
import sys
import numpy as np
import random
from omegaconf import DictConfig

from rovi_aug.mods.base_mod import BaseMod, add_obs_key

class AugMergeMod(BaseMod):
    robot_aug_imgs_input_key = ""
    inpainted_background_input_key = ""
    merged_output_key = ""

    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        img_size = features["steps"]["observation"][AugMergeMod.inpainted_background_input_key].shape[0]
        return add_obs_key(features, AugMergeMod.merged_output_key, tfds.features.Tensor(shape=(img_size, img_size, 3), dtype=tf.uint8))

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:

        def augment_view(step):            
            step["observation"][AugMergeMod.merged_output_key] = tf.numpy_function(AugMergeMod.merge_images, [step["observation"][AugMergeMod.inpainted_background_input_key], step["observation"][AugMergeMod.robot_aug_imgs_input_key]], tf.uint8)
            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].map(augment_view)
            return episode

        return ds.map(episode_map_fn)

    @classmethod
    def load(cls, cfg: DictConfig):
        """
        Uses information from the config file to load the mod.
        """
        AugMergeMod.robot_aug_imgs_input_key = cfg.aug_merge.robot_aug_imgs_input_key
        AugMergeMod.inpainted_background_input_key = cfg.aug_merge.inpainted_background_input_key
        AugMergeMod.merged_output_key = cfg.aug_merge.merged_output_key

    @staticmethod
    def change_brightness(img, mean_value=100, mask=None, randomness=0):
        """
        Augments the brightness so that the resulting images can be more in distribution with real images.
        Args:
            img: np.ndarray of shape (H, W, 3)
            mean_value: int, the mean value of the brightness augmentation
            mask: np.ndarray of shape (H, W), the mask to apply the brightness augmentation
            randomness: int, the randomness of the brightness augmentation

        Returns:
            np.ndarray of shape (H, W, 3), the augmented image via brightness
        """
        # Lazy load
        import cv2
        value = mean_value + random.randint(-randomness, randomness)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        if mask is None:
            mask = np.ones_like(v)
        else:
            mask = mask.squeeze()
        
        if value > 0:
            lim = 255 - value
            v[(v > lim) & (mask == 1)] = 255
            v[(v <= lim) & (mask == 1)] += value
        else:
            lim = -value
            v[(v < lim) & (mask == 1)] = 0
            v[(v >= lim) & (mask == 1)] -= lim

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    @staticmethod
    def segment_robot(robot_image):
        """
        Extracts the robot from the white background using a combination of color and edge detection.

        Args:
            robot_image: np.ndarray of shape (H, W, 3)
        Returns:
            np.ndarray of shape (H, W), which is the robot masked out from the background
        """

        # Lazy load
        import cv2
        hsv = cv2.cvtColor(robot_image, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([0, 0, 230])
        upper_bound = np.array([180, 50, 255])
        
        color_mask = cv2.inRange(hsv, lower_bound, upper_bound)
        robot_mask = cv2.bitwise_not(color_mask)
        
        edges = cv2.Canny(robot_image, 100, 200)
        combined_mask = cv2.bitwise_or(robot_mask, edges)
        
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        eroded_mask = cv2.erode(combined_mask, kernel=np.ones((5, 5), np.uint8), iterations=1)
        final_mask = cv2.GaussianBlur(eroded_mask, (3, 3), 0)
        
        return final_mask

    @staticmethod
    def merge_images(background, robot_image):
        """
        Performs the actual merging of the robot with the inpainted background.

        Args:
            background: np.ndarray of shape (H, W, 3)
            robot_image: np.ndarray of shape (H, W, 3)
        """
        # Lazy load
        from PIL import Image
        background = background
        background = Image.fromarray(background)
        
        # This robot mask refers to getting the robot from the completely white background
        robot_mask = AugMergeMod.segment_robot(robot_image)
        
        brightened_robot_image_augmented = AugMergeMod.change_brightness(robot_image, mean_value=50, randomness=30)
        
        robot_mask = Image.fromarray(robot_mask).convert("L")

        robot_image_augmented = Image.fromarray(brightened_robot_image_augmented).convert('RGBA')

        if robot_mask.size != robot_image_augmented.size:
            robot_mask = robot_mask.resize(robot_image_augmented.size, Image.ANTIALIAS)
        
        # Save augmented images
        augmented_background = background.copy()
        augmented_background.paste(robot_image_augmented, (0, 0), robot_mask)
        return np.array(augmented_background)