import tensorflow as tf
import tensorflow_datasets as tfds
import importlib
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig

from rovi_aug.mods.base_mod import BaseMod, add_obs_key

class VideoInpaintMod(BaseMod):
    batch_size = 400
    device = "cuda:0"
    image_input_key = ""
    mask_input_key = ""
    image_output_key = ""

    neighbor_stride = 5
    size = (256, 256)
    num_ref = -1
    ref_length = 10

    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        img_size = features["steps"]["observation"][VideoInpaintMod.image_input_key].shape[0]
        return add_obs_key(features, VideoInpaintMod.image_output_key, tfds.features.Tensor(shape=(img_size, img_size, 3), dtype=tf.uint8))

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        def inpaint_background(step):
            def process_images(trajectory_images, masked_images):
                try:
                    non_stacked_imgs = VideoInpaintMod.perform_inpainting(trajectory_images, masked_images)
                    return non_stacked_imgs
                except Exception as e:
                    print(e)
                    return trajectory_images
                # return np.stack(non_stacked_imgs, axis=0).astype(np.uint8)

            step["observation"][VideoInpaintMod.image_output_key] = tf.numpy_function(process_images, [step["observation"][VideoInpaintMod.image_input_key], step["observation"][VideoInpaintMod.mask_input_key]], tf.uint8)
            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].batch(VideoInpaintMod.batch_size).map(inpaint_background).unbatch()
            return episode

        return ds.map(episode_map_fn)

    @classmethod
    def load(cls, cfg: DictConfig):
        """
        Uses information from the config file to load the mod.
        """
        VideoInpaintMod.device = cfg.device
        VideoInpaintMod.batch_size = cfg.video_inpaint.batch_size
        VideoInpaintMod.image_input_key = cfg.video_inpaint.image_input_key
        VideoInpaintMod.mask_input_key = cfg.video_inpaint.mask_input_key
        VideoInpaintMod.image_output_key = cfg.video_inpaint.image_output_key
        
        video_inpaint_checkpoint_path = cfg.video_inpaint.video_inpaint_checkpoint_path

        VideoInpaintMod.net = importlib.import_module('model.e2fgvi_hq')
        VideoInpaintMod.model = VideoInpaintMod.net.InpaintGenerator().to(VideoInpaintMod.device)
        
        data = torch.load(video_inpaint_checkpoint_path, map_location=VideoInpaintMod.device)
        VideoInpaintMod.model.load_state_dict(data)
        VideoInpaintMod.model.eval()

    @staticmethod
    def process_masks(masks):
        from PIL import Image
        import cv2

        new_masks = []
        for i in range(masks.shape[0]):
            m = Image.fromarray(masks[i])
            m = m.resize(VideoInpaintMod.size, Image.NEAREST)
            m = np.array(m.convert('L'))
            m = np.array(m > 0).astype(np.uint8)
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=4)
            new_masks.append(Image.fromarray(m * 255))
        return new_masks

    @staticmethod
    def get_ref_index(f, neighbor_ids, length):
        ref_index = []
        if VideoInpaintMod.num_ref == -1:
            for i in range(0, length, VideoInpaintMod.ref_length):
                if i not in neighbor_ids:
                    ref_index.append(i)
        else:
            start_idx = max(0, f - VideoInpaintMod.ref_length * (VideoInpaintMod.num_ref // 2))
            end_idx = min(length, f + VideoInpaintMod.ref_length * (VideoInpaintMod.num_ref // 2))
            for i in range(start_idx, end_idx + 1, VideoInpaintMod.ref_length):
                if i not in neighbor_ids:
                    if len(ref_index) > VideoInpaintMod.num_ref:
                        break
                    ref_index.append(i)
        return ref_index

    @staticmethod
    def perform_inpainting(imgs, masks):  # M x 3 x H x W, M x 1 x H x W
        with torch.no_grad():
            frames = imgs
            imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().unsqueeze(0) / 255.0
            imgs = imgs * 2 - 1
            processed_masks = VideoInpaintMod.process_masks(masks)
            masks = VideoInpaintMod.to_tensors()(processed_masks).unsqueeze(0)
            masks = masks.float()  # Ensure masks are float

            cpu_masks = masks.cpu().numpy().squeeze().astype(np.uint8)[..., None]
            
            imgs, masks = imgs.to(VideoInpaintMod.device), masks.to(VideoInpaintMod.device)
            video_length = imgs.shape[1]
            h, w = imgs.shape[3], imgs.shape[4]
            # comp_frames = [None] * video_length
            put_frame = [False] * video_length
            comp_frames = np.zeros((video_length, h, w, 3))

            for f in tqdm(range(0, video_length, VideoInpaintMod.neighbor_stride)):
                neighbor_ids = [i for i in range(max(0, f - VideoInpaintMod.neighbor_stride), min(video_length, f + VideoInpaintMod.neighbor_stride + 1))]
                ref_ids = VideoInpaintMod.get_ref_index(f, neighbor_ids, video_length)
                selected_imgs = imgs[:1, neighbor_ids + ref_ids]
                selected_masks = masks[:1, neighbor_ids + ref_ids]
                masked_imgs = selected_imgs * (1 - selected_masks)
                mod_size_h = 60
                mod_size_w = 108
                h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
                w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
                masked_imgs = torch.cat([masked_imgs, torch.flip(masked_imgs, [3])], 3)[:, :, :, :h + h_pad, :]
                masked_imgs = torch.cat([masked_imgs, torch.flip(masked_imgs, [4])], 4)[:, :, :, :, :w + w_pad]
                pred_imgs, _ = VideoInpaintMod.model(masked_imgs, len(neighbor_ids))
                pred_imgs = pred_imgs[:, :, :h, :w]
                pred_imgs = (pred_imgs + 1) / 2
                pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = pred_imgs[i] * cpu_masks[idx] + frames[idx] * (1 - cpu_masks[idx])
                    if put_frame[idx]:
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                    else:
                        comp_frames[idx] = img
                        put_frame[idx] = True

            return comp_frames.astype(np.uint8)
    
    @staticmethod
    def to_tensors():
        from torchvision import transforms
        return transforms.Compose([Stack(), ToTorchFormatTensor()])


# Borrowed directly from the e2fgvi repo, but since the module is not well defined, copying it here
class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        mode = img_group[0].mode
        if mode == '1':
            img_group = [img.convert('L') for img in img_group]
            mode = 'L'
        if mode == 'L':
            return np.stack([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif mode == 'RGB':
            if self.roll:
                return np.stack([np.array(x)[:, :, ::-1] for x in img_group],
                                axis=2)
            else:
                return np.stack(img_group, axis=2)
        else:
            raise NotImplementedError(f"Image mode {mode}")


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # numpy img: [L, C, H, W]
            img = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(
                pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255) if self.div else img.float()
        return img