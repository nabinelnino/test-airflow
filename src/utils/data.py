import gzip
import os.path
import pickle

import numpy as np
from src.utils.cluster import cluster_leader_from_array


class AircheckDataset:
    def __init__(self, data: np.ndarray, y: np.ndarray, fp: str, groups: np.ndarray = None):
        self.fp = fp
        self.data = data
        self.y = np.array(y)
        self.groups = groups

        assert self.data.shape[0] == self.y.shape[0]

    def generate_groups(self):
        self.groups = cluster_leader_from_array(self.data)
        assert len(self.groups) == self.data.shape[0]

    def save(self, path: str):
        _name = f"aircheck_dataset_{self.fp}"
        with gzip.open(os.path.join(path, f"X_{self.fp}.pkl.gz"), "wb") as f:
            pickle.dump(self.data, f)
        with gzip.open(os.path.join(path, f"y.pkl.gz"), "wb") as f:
            pickle.dump(self.y, f)
        if self.groups is not None:
            with gzip.open(os.path.join(path, f"cluster.pkl.gz"), "wb") as f:
                pickle.dump(self.groups, f)
        return path

    @staticmethod
    def load(fp: str, path: str):
        if not os.path.exists(os.path.join(path, f"X_{fp}.pkl.gz")) or os.path.exists(os.path.join(path, f"y.pkl.gz")):
            raise FileNotFoundError("cannot find required files to load from dataset directory")

        with gzip.open(os.path.join(path, f"X_{fp}.pkl.gz"), "rb") as f:
            data = pickle.load(f)
        with gzip.open(os.path.join(path, f"y.pkl.gz"), "rb") as f:
            y = pickle.load(f)
        if os.path.exists(os.path.join(path, f"cluster.pkl.gz")) is not None:
            with gzip.open(os.path.join(path, f"cluster.pkl.gz"), "rb") as f:
                groups = pickle.load(f)
        else:
            groups = None

        return AircheckDataset(data, y, fp, groups)
