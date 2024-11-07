import tensorflow as tf
import tensorflow_datasets as tfds
from importlib import import_module
import torch
import random
import numpy as np
import sys
from omegaconf import DictConfig

from rovi_aug.mods.base_mod import BaseMod, add_obs_key

class RobotMaskMod(BaseMod):
    mask_generator = None
    batch_size = 256
    device = "cuda:0"
    image_input_key = "front_rgb"
    mask_output_key = "masks"
    masked_images_output_key = "masked_imgs"

    mask_args = dict()
    model = None
    multimask_output = False

    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        img_size = features["steps"]["observation"][RobotMaskMod.image_input_key].shape[0]
        new_feature_tensor_type = tfds.features.Tensor(shape=(img_size, img_size, 3), dtype=tf.uint8)

        # Adds the mask and masked image features to the output observations
        first_added_feature_dict = add_obs_key(features, RobotMaskMod.mask_output_key, new_feature_tensor_type)
        return add_obs_key(first_added_feature_dict, RobotMaskMod.masked_images_output_key, new_feature_tensor_type)

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        def generate_masks(step):
            def process_images(trajectory_images):
                masked_images, output_masks = RobotMaskMod.inference(torch.from_numpy(trajectory_images).permute(0, 3, 1, 2))
                masked_images *= 255
                masked_images = masked_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                output_masks = output_masks.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                return np.concatenate([masked_images, output_masks], axis=-1)

            processed_output = tf.numpy_function(process_images, [step["observation"][RobotMaskMod.image_input_key]], tf.uint8)
            step["observation"][RobotMaskMod.masked_images_output_key] = processed_output[..., :3]
            step["observation"][RobotMaskMod.mask_output_key] = processed_output[..., 3:]

            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].batch(RobotMaskMod.batch_size).map(generate_masks).unbatch()
            return episode

        return ds.map(episode_map_fn)
    
    @classmethod
    def load(cls, cfg: DictConfig):
        """
        Uses information from the config file to load the mod.
        """
        RobotMaskMod.device = cfg.device
        RobotMaskMod.batch_size = cfg.robot_mask.batch_size
        RobotMaskMod.image_input_key = cfg.robot_mask.image_input_key
        RobotMaskMod.mask_output_key = cfg.robot_mask.mask_output_key
        RobotMaskMod.masked_images_output_key = cfg.robot_mask.masked_images_output_key

        sam_checkpoint_path = cfg.robot_mask.sam_checkpoint_path
        sam_package_path = cfg.robot_mask.sam_package_path
        sys.path.append(sam_package_path)
        sam_lora_checkpoint_path = cfg.robot_mask.sam_lora_checkpoint_path

        RobotMaskMod.mask_args = {
            "num_classes": 1,
            "img_size": 256,
            "input_size": 256,
            "seed": 1234,
            "deterministic": 1,
            "ckpt": sam_checkpoint_path,
            "lora_ckpt": sam_lora_checkpoint_path,
            "vit_name": "vit_h",
            "rank": 4,
            "module": "sam_lora_image_encoder",
            "mask_threshold": 0.5,
            "device": RobotMaskMod.device,
        }
        RobotMaskMod.setup_seed()
        RobotMaskMod.setup_model()

    @staticmethod
    def inference(images):
        """
        Generates masked out iamges and the corresponding robot masks for the input images.
        """
        with torch.no_grad():
            resulting_size = (RobotMaskMod.mask_args['img_size'], RobotMaskMod.mask_args['img_size'])
            images = torch.nn.functional.upsample_bilinear(images.to(RobotMaskMod.mask_args['device']).float() / 1.0, size=resulting_size)
            outputs = RobotMaskMod.model(images, RobotMaskMod.multimask_output, RobotMaskMod.mask_args['img_size'])

            output_masks = outputs['masks']
            output_masks = torch.nn.functional.upsample_bilinear(output_masks.float(), size=resulting_size)
            output_masks = (output_masks[:, 1, :, :] > RobotMaskMod.mask_args['mask_threshold']).unsqueeze(1)

            masked_image = (images / 255.0) * output_masks

            output_masks = output_masks.expand_as(images)
            masked_image[output_masks == 0] = 1
            return masked_image, output_masks
        
    @staticmethod
    def setup_model():
        # Lazy load

        from segment_anything import sam_model_registry
        sam, _ = sam_model_registry[RobotMaskMod.mask_args['vit_name']](
            image_size=RobotMaskMod.mask_args['img_size'],
            num_classes=RobotMaskMod.mask_args['num_classes'],
            checkpoint=RobotMaskMod.mask_args['ckpt']
        )
        sam = sam.to(RobotMaskMod.mask_args['device'])
        pkg = import_module(RobotMaskMod.mask_args['module'])
        RobotMaskMod.model = pkg.LoRA_Sam(sam, RobotMaskMod.mask_args['rank'])

        assert RobotMaskMod.mask_args['lora_ckpt'] is not None
        RobotMaskMod.model.load_lora_parameters(RobotMaskMod.mask_args['lora_ckpt'])
        RobotMaskMod.model = RobotMaskMod.model.to(RobotMaskMod.mask_args['device'])

        RobotMaskMod.multimask_output = RobotMaskMod.mask_args['num_classes'] > 1
        RobotMaskMod.model.eval()

    @staticmethod
    def setup_seed():
        import torch.backends.cudnn as cudnn
        if not RobotMaskMod.mask_args['deterministic']:
            cudnn.benchmark = True
            cudnn.deterministic = False
        else:
            cudnn.benchmark = False
            cudnn.deterministic = True
        seed = RobotMaskMod.mask_args['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)