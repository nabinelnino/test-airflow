from copy import deepcopy

import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator

from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit
from sklearn.feature_selection import VarianceThreshold

from src.utils.fp import get_fp_func


class AirCheckModel(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model: ClassifierMixin, fp: str):
        self.model = deepcopy(base_model)
        self.variance_threshold = VarianceThreshold(0.01)
        self.fp = fp

    def fit(self, X, y, groups = None, subsample: float = None, subsample_on_group: bool = True):
        if subsample is not None:
            if subsample_on_group:
                if groups is None:
                    raise RuntimeError("cannot subsample on groups when groups is `None`")
                s = GroupShuffleSplit(n_splits=1, test_size=subsample)
                train_idx, _ = s.split(X, y, groups=groups).__next__()
            else:
                s = ShuffleSplit(n_splits=1, test_size=subsample)
                train_idx, _ = s.split(X, y).__next__()

        X = self.variance_threshold.fit_transform(X)
        self.model.fit(X, y)

    def predict_smiles(self, smiles, batch_size=1000):
        preds = np.array([])
        for batch in range(0, len(smiles), batch_size):
            batch_smiles = smiles[batch:batch + batch_size]
            batch_fps = get_fp_func(self.fp).transform(batch_smiles)
            batch_preds = self.predict(batch_fps)
            preds = np.concatenate([preds, batch_preds])

    def predict(self, X):
        X = self.variance_threshold.transform(X)
        return self.model.predict(X)

    def predict_proba_smiles(self, smiles, batch_size=1000):
        preds = np.array([])
        for batch in range(0, len(smiles), batch_size):
            batch_smiles = smiles[batch:batch + batch_size]
            batch_fps = get_fp_func(self.fp).transform(batch_smiles)
            batch_preds = self.predict_proba(batch_fps)
            preds = np.concatenate([preds, batch_preds])

    def predict_proba(self, X):
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError('this model has no predict_proba method')
        X = self.variance_threshold.transform(X)
        return self.model.predict_proba(X)
