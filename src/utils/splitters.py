from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, GroupShuffleSplit

from src.utils.cluster import cluster_scaffold, cluster_leader_from_array


class StratifiedScaffoldKFold(StratifiedGroupKFold):
    """
    Generate cross-validation splits using a Stratified Scaffold approach
    NOT recommend for model validation
    """
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state=None):
        super().__init__(n_splits, shuffle, random_state)

    def split(self, X, y=None, groups=None):
        if groups is None:
            groups = cluster_scaffold(X)
        for train, test, in super().split(X, y, groups):
            yield train, test


class StratifiedTanimotoClusterKFold(StratifiedGroupKFold):
    """
    Generate cross-validation splits using a Stratified Chemical similarity approach
    RECOMMENDED for model validation
    """
    def __init__(self, n_splits: int = 5, thresh: float = 0.65, shuffle: bool = True, random_state=None):
        super().__init__(n_splits, shuffle, random_state)
        self.thresh = thresh

    def split(self, X, y=None, groups=None):
        if groups is None:
            groups = cluster_leader_from_array(X, thresh=self.thresh)
        for train, test, in super().split(X, y, groups):
            yield train, test


class ScaffoldKFold(GroupKFold):
    """
    Generate cross-validation splits using a KFold Scaffold approach
    NOT recommend for model validation
    """
    def __init__(self, n_splits: int = 5):
        super().__init__(n_splits)

    def split(self, X, y=None, groups=None):
        if groups is None:
            groups = cluster_scaffold(X)
        for train, test, in super().split(X, y, groups):
            yield train, test


class TanimotoClusterKFold(GroupKFold):
    """
    Generate cross-validation splits using a KFold Chemical similarity approach
    RECOMMENDED for model validation
    """
    def __init__(self, n_splits: int = 5, thresh: float = 0.65):
        super().__init__(n_splits)
        self.thresh = thresh

    def split(self, X, y=None, groups=None):
        if groups is None:
            groups = cluster_leader_from_array(X, thresh=self.thresh)
        for train, test, in super().split(X, y, groups):
            yield train, test


class ShuffleScaffoldKFold(GroupShuffleSplit):
    """
    Generate cross-validation splits using a Shuffled Scaffold approach
    NOT recommend for model validation
    """
    def __init__(self, test_size: float = 0.2, random_state=None):
        super().__init__(test_size=test_size, random_state=random_state)

    def split(self, X, y=None, groups=None):
        if groups is None:
            groups = cluster_scaffold(X)
        for train, test, in super().split(X, y, groups):
            yield train, test


class ShuffleTanimotoClusterKFold(GroupShuffleSplit):
    """
    Generate cross-validation splits using a Shuffled Chemical similarity approach
    RECOMMENDED for model validation
    """
    def __init__(self, test_size: float = 0.2, thresh: float = 0.65, random_state=None):
        super().__init__(test_size=test_size, random_state=random_state)
        self.thresh = thresh

    def split(self, X, y=None, groups=None):
        if groups is None:
            groups = cluster_leader_from_array(X, thresh=self.thresh)
        for train, test, in super().split(X, y, groups):
            yield train, test
