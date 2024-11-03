from abc import abstractmethod
from omegaconf import DictConfig
import rlds_dataset_mod.mod_functions as mod_fns

class BaseMod(mod_fns.TfdsModFunction):
    """
    Interfaces with the RLDS code, except also provides a load function
    for setting up any necessary configured state.
    """

    @classmethod
    @abstractmethod
    def load(cfg: DictConfig):
        """
        Uses information from the config file to load the mod.
        """
        pass