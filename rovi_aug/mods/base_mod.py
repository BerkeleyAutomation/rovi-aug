from abc import abstractmethod
from omegaconf import DictConfig
import rlds_dataset_mod.mod_functions as mod_fns
import tensorflow_datasets as tfds

def add_obs_key(features, obs_key, feature_size):
    """Utility function to only modify keys in observation dict."""
    observations = {
                    key: features["steps"]["observation"][key]
                    for key in features["steps"]["observation"].keys()
                }
    observations[obs_key] = feature_size

    return tfds.features.FeaturesDict(
        {
            "steps": tfds.features.Dataset(
                {
                    "observation": tfds.features.FeaturesDict(observations),
                    **{
                        key: features["steps"][key]
                        for key in features["steps"].keys()
                        if key not in ("observation",)
                    },
                }
            ),
            **{key: features[key] for key in features.keys() if key not in ("steps",)},
        }
    )

class BaseMod(mod_fns.TfdsModFunction):
    """
    Interfaces with the RLDS code, except also provides a load function
    for setting up any necessary configured state.
    """

    @classmethod
    @abstractmethod
    def load(cls, cfg: DictConfig):
        """
        Uses information from the config file to load the mod.
        """
        pass