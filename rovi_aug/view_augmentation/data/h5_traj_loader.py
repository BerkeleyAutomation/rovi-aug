from typing import List
import h5py
from robomimic.scripts.config_gen.helper import scan_datasets

class H5TrajLoader:

    def __init__(self, 
                 trajectory_dataset_folder: str, 
                 partition_idx: int, 
                 num_partitions: int) -> None:
        """
        Initializes a trajectory loader
        :param traj_files: A list of trajectory files to load
        """
        all_traj_files = sorted(scan_datasets(trajectory_dataset_folder))
        self.traj_files = all_traj_files[partition_idx::num_partitions]

    def __iter__(self):
        for traj_file in self.traj_files:
            # Load the trajectory file
            with h5py.File(traj_file, 'r+') as f:
                yield f['observation']['camera']['image']
    
    def __len__(self):
        return len(self.traj_files)