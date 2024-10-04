import gzip
import os.path
import pickle

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import Mol
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.SimDivFilters import rdSimDivPickers
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

from src.utils.utils import to_mol


def generate_scaffold(mol: str or Mol or object or None, include_chirality: bool = False) -> str or None:
    """
    Takes a mol (or SMILES, or object with `.smiles` attribute) and returns its Murko Scaffold as SMILES

    Parameters
    ----------
    mol: str or Mol or object
        Mol (or SMILES) to get scaffold for
    include_chirality: bool, default False
        include stereochemistry in scaffold

    Returns
    -------
    smiles: str or None
        SMILES of scaffold, None if Mol or SMILES is None or invalid
    """
    to_mol(mol)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)


def cluster_scaffold(smis: list[Mol or str]):
    """
    Generate a cluster id map for a list of SMILES such that each cluster only has one murko scaffold.

    Notes
    -----
    passed smiles can be Mol objects for just the raw text SMILES

    Parameters
    ----------
    smis: list[Mol or str]
        chemicals to generate cluster index for

    Returns
    -------
    cluster_ids: np.ndarray[int]
        an array of cluster ids, index mapped to the passed smis

    """
    _scaffolds = {}
    _ticker = 0
    _idxs = []
    for smi in smis:
        _scaffold = generate_scaffold(smi)
        _idx = _scaffolds.get(_scaffold, None)
        if _idx is None:
            _scaffolds[_scaffold] = _ticker
            _idx = _ticker
            _ticker += 1
        _idxs.append(_idx)
    return np.array(_idxs)


def cluster_leader_smiles(smis, thresh: float = 0.65, use_tqdm: bool = False):
    """
    Generate a cluster id map for a list of SMILES such that each cluster centroid has a tanimoto similarity below the
    passed threshold. Each chemical that is not a centroid is a member of the cluster that it shares the highest
    similarity to.

    This means that not every cluster will have a total separation of 0.35 tanimoto distance.

    Notes
    -----
    passed smiles can be Mol objects for just the raw text SMILES

    Parameters
    ----------
    smis: list[Mol or str]
        chemicals to generate cluster index for
    thresh: float, default 0.65
        the tanimoto distance (1-similarity) that you want centroid to have
    use_tqdm: bool, default False
        track clustering progress with a tqdm progress bar

    Returns
    -------
    cluster_ids: np.ndarray[int]
        an array of cluster ids, index mapped to the passed smis

    """
    _fps = [GetMorganFingerprintAsBitVect(to_mol(smi)) for smi in smis]
    lp = rdSimDivPickers.LeaderPicker()

    _centroids = lp.LazyBitVectorPick(_fps, len(_fps), thresh)
    _centroid_fps = [_fps[i] for i in _centroids]

    _cluster_ids = []
    for _fp in tqdm(_fps, disable=not use_tqdm, desc="assigning SMILES to clusters"):
        sims = BulkTanimotoSimilarity(_fp, _centroid_fps)
        _cluster_ids.append(np.argmax(sims))
    return np.array(_cluster_ids)


def cluster_leader_from_array(X, thresh: float = 0.65, use_tqdm: bool = False):
    """
    Generate a cluster id map for already featurized array such that each cluster centroid has a tanimoto similarity
    below the passed threshold. Each chemical that is not a centroid is a member of the cluster that it shares the
     highest similarity to.

    This means that not every cluster will have a total separation of 0.35 tanimoto distance.

    Notes
    -----
    passed smiles can be Mol objects for just the raw text SMILES

    Parameters
    ----------
    smis: list[Mol or str]
        chemicals to generate cluster index for
    thresh: float, default 0.65
        the tanimoto distance (1-similarity) that you want centroid to have
    use_tqdm: bool, default False
        track clustering progress with a tqdm progress bar

    Returns
    -------
    cluster_ids: np.ndarray[int]
        an array of cluster ids, index mapped to the passed smis

    """
    _fps = [DataStructs.CreateFromBitString("".join(["1" if __ > 0 else "0" for __ in _])) for _ in X]
    lp = rdSimDivPickers.LeaderPicker()

    _centroids = lp.LazyBitVectorPick(_fps, len(_fps), thresh)
    _centroid_fps = [_fps[i] for i in _centroids]

    _cluster_ids = []
    for _fp in tqdm(_fps, disable=not use_tqdm, desc="assigning SMILES to clusters"):
        sims = BulkTanimotoSimilarity(_fp, _centroid_fps)
        _cluster_ids.append(np.argmax(sims))
    return np.array(_cluster_ids)


def generate_custer_ids_for_aircheck_array(X: np.ndarray or str, method: str = "tanimoto", save_to: str = None,
                                           use_tqdm: bool = True):
    """
    Generate the cluster ids for an AIRCHECK datafile or array
    Parameters
    ----------
    X: np.ndarray or str
        AIRCHECK fingerprint array
        if a str, will try and read in the file as a .pkl.gzip file
    method: str, default "tanimoto"
        which clustering method to use
        right now, only tanimoto exists
    save_to: str, default None
        if not None, will save cluster file as cluster.pkl.gz
    use_tqdm: bool, default True
        track clustering progress with a tqdm progress bar

    Returns
    -------
    cluster_ids: np.ndarray[int]
        the cluster_ids index mapped to the rows of X
    """

    if isinstance(X, str):
        with gzip.open(X, "rb") as f:
            X = pickle.load(f)

    if method == "tanimoto":
        cluster_ids = cluster_leader_smiles(X, use_tqdm=use_tqdm)
    else:
        raise ValueError("method must be tanimoto (future updates may add more")

    if save_to is not None:
        with gzip.open(os.path.join(save_to, "cluster.pkl.gz"), "wb") as f:
            pickle.dump(cluster_ids, f)

    return cluster_ids
