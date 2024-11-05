"""Modifies TFDS dataset with a map function, updates the feature definition and stores new dataset."""
from functools import partial

import omegaconf
import argparse
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

import tensorflow_datasets as tfds

import rlds_dataset_mod.mod_functions as mod_fns
from rlds_dataset_mod.mod_functions import TFDS_MOD_FUNCTIONS
from rovi_aug.data_processing.multithreaded_adhoc_tfds_builder import (
    MultiThreadedAdhocDatasetBuilder,
)
from rovi_aug.mods.base_mod import BaseMod 

def mod_features(features, mods):
    """Modifies feature dict."""
    for mod in mods:
        features = TFDS_MOD_FUNCTIONS[mod].mod_features(features)
    return features


def mod_dataset_generator(builder, split, mods):
    """Modifies dataset features."""
    ds = builder.as_dataset(split=split)
    for mod in mods:
        ds = TFDS_MOD_FUNCTIONS[mod].mod_dataset(ds)
    for episode in tfds.core.dataset_utils.as_numpy(ds):
        yield episode

def load_mods(cfg: omegaconf.DictConfig, mods):
    """
    Loads the mods if necessary.
    """
    for mod in mods:
        mod_fn = TFDS_MOD_FUNCTIONS[mod]
        if isinstance(mod_fn, BaseMod):
            mod_fn.load(cfg)

def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mods", nargs="+", help="List of augmentation functions. Valid functions include" +
                         "robot_mask, robot_to_robot, video_inpaint, aug_merge, view_augmentation. WARNING: This is only tested with 1 input at a time.")
    parser.add_argument("--conf", help="File path to the pipeline configuration file.")
    args = parser.parse_args()

    cfg = omegaconf.OmegaConf.load(args.conf)
    mods = args.mods
    load_mods(cfg, mods)
    builder = tfds.builder(cfg.dataset, data_dir=cfg.data_dir)

    features = mod_features(builder.info.features, mods)
    print("############# Target features: ###############")
    print(features)
    print("##############################################")
    assert cfg.data_dir != cfg.target_dir   # prevent overwriting original dataset

    mod_dataset_builder = MultiThreadedAdhocDatasetBuilder(
        name=cfg.dataset,
        version=builder.version,
        features=features,
        split_datasets={split: builder.info.splits[split] for split in builder.info.splits},
        config=builder.builder_config,
        data_dir=cfg.target_dir,
        description=builder.info.description,
        generator_fcn=partial(mod_dataset_generator, builder=builder, mods=cfg.mods),
        n_workers=cfg.n_workers,
        max_episodes_in_memory=cfg.max_episodes_in_memory,
    )
    mod_dataset_builder.download_and_prepare()


if __name__ == "__main__":
    main()
