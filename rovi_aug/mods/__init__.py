from rlds_dataset_mod.mod_functions import TFDS_MOD_FUNCTIONS

from rovi_aug.mods.merge_mod import AugMergeMod
from rovi_aug.mods.robot_mask_mod import RobotMaskMod
from rovi_aug.mods.video_inpaint_mod import VideoInpaintMod
from rovi_aug.mods.robot_to_robot_mod import R2RMod
from rovi_aug.mods.view_augmentation_mod import ViewAugmentationMod

ROVI_AUG_MODS = {
    "robot_mask": RobotMaskMod,
    "robot_to_robot": R2RMod,
    "video_inpaint": VideoInpaintMod,
    "aug_merge": AugMergeMod,
    "view_augmentation": ViewAugmentationMod
}

TFDS_MOD_FUNCTIONS.update(ROVI_AUG_MODS)