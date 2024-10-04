import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import gzip
from typing import Union, List, Optional

from src.utils.utils import to_list
from src.utils.data import AircheckDataset
import glob

import logging

HITGEN_FPS_COLS = {
    'ECFP4',
    'ECFP6',
    'FCFP4',
    'FCFP6',
    'MACCS',
    'RDK',
    'AVALON',
    'ATOMPAIR',
    'TOPTOR'
}


HITGEN_FPS_COLS_MAP = {
    'ECFP4': "HitGenECFP4",
    'ECFP6': "HitGenECFP6",
    'FCFP4': "HitGenFCFP4",
    'FCFP6': "HitGenFCFP6",
    'MACCS': "HitGenMACCS",
    'RDK': "HitGenRDK",
    'AVALON': "HitGenAvalon",
    'ATOMPAIR': "HitGenAtomPair",
    'TOPTOR': "HitGenTopTor"
}

HITGEN_FPS_COLS_MAP_BINARY = {
    'ECFP4': "HitGenBinaryECFP4",
    'ECFP6': "HitGenBinaryECFP6",
    'FCFP4': "HitGenBinaryFCFP4",
    'FCFP6': "HitGenBinaryFCFP6",
    'AVALON': "HitGenBinaryAvalon",
    'ATOMPAIR': "HitGenBinaryAtomPair",
    'TOPTOR': "HitGenBinaryTopTor"
}

XCHEM_FPS_COLS = {
    'ECFP4',
    'ECFP6',
    'FCFP4',
    'FCFP6',
    'MACCS'
}

XCHEM_FPS_COLS_MAP = {
    'ECFP4': "XChemBinaryECFP4",
    'ECFP6': "XChemBinaryECFP6",
    'FCFP4': "XChemBinaryFCFP4",
    'FCFP6': "XChemBinaryECFP4",
    'MACCS': "XChemMACCS"
}


def read_aircheck_file(
    path: str,
    company: str,
    fps: Union[str, List[str]],
    file_format: Optional[str] = None,
    save_to: Optional[str] = None,
    binarize: bool = False
):
    """
    Reads in a AIRCHECK data file downloaded from the AIRCHECK website and loads in the fingerprint data and labels,
    returning them as arrays

    Optionally, can also save these arrays to a .pkl.gz file

    Parameters
    ----------
    path: str
        path to tsv file
    company: str
        company name for the file (for processing different file types)
    fps: str or list[str]
        the fingerprint types you want to collect
    save_to: str, default None
        if not None, the path to where to save the data
        will make a X_[FPS].pkl.gz file for each fingerprint and a single y.pkl.gz
    binarize: bool, default False
        if company is "hitgen", binarize the fingerprint and return it as a binary FP

    Returns
    -------
    dataset(s): list[AircheckDataSet]
        a list of AirCheckDataSet objects with the corresponding FpFunc and X, y data
    """

    if company == "hitgen":

        valid_fps_col = set(to_list(fps)).intersection(HITGEN_FPS_COLS)

        X, y = _read_hitgen(path, valid_fps_col, True)
    elif company == "xchem":
        print("fingerprints", fps)
        print(type(fps))
        valid_fps_col = set(to_list(fps)).intersection(XCHEM_FPS_COLS)
        print("valid fps", valid_fps_col)
        if file_format == "parquet":
            X, y = _read_xchem_parquet(path, valid_fps_col)
        else:
            X, y = _read_xchem(path, valid_fps_col)
    else:
        raise ValueError(f"Company {company} not recognized")

    if len(X) == 0:
        raise ValueError(f"no valid fps cols were passed: {fps}")

    if save_to is not None:
        for fp, arr in X.items():
            with gzip.open(os.path.join(save_to, f"X_{fp}.pkl.gz"), 'wb') as f:
                pickle.dump(arr, f)
        with gzip.open(os.path.join(save_to, f"y.pkl.gz"), 'wb') as f:
            pickle.dump(arr, y)

    datasets = []
    for fp, arr in X.items():

        datasets.append(AircheckDataset(arr, y, fp))

    return datasets


# TODO get real column name for y
def _read_hitgen(path, fps, binarize: bool = False):
    logging.info(f"Starting to read file: {path}")
    logging.info(f"File size: {os.path.getsize(path)} bytes")
    X = {}
    y = []

    for fp in tqdm(fps):
        if binarize:
            fp_key = HITGEN_FPS_COLS_MAP_BINARY.get(fp, None)
            if fp_key is None:
                raise ValueError(f"cannot make {fp} binary for HitGen")
        else:
            fp_key = HITGEN_FPS_COLS_MAP_BINARY.get(fp)
        y = []
        _x = []

        with gzip.open(path, 'rt', newline='', encoding='utf-8') as f:
            header = f.readline().strip().split("\t")
            fp_idx = header.index(fp)
            for line in tqdm(f):
               
                if line.strip() == "":
                    continue
                splits = line.strip().split("\t")
                y.append(int(splits[2]))
                if binarize:
                    _x.append(
                        [1 if int(_) > 0 else 0 for _ in splits[fp_idx].split(",")])
                else:
                    _x.append([int(_) for _ in splits[fp_idx].split(",")])
            X[fp_key] = np.array(_x)
    print("Type of X and Y", type(X), type(y))
    logging.info(
        f"Finished processing. X size: {sum(x.size for x in X.values())}, y size: {len(y)}")

    return X, y


def _read_xchem(path, fps):
   
    X = {}
    y = []
    for fp in tqdm(fps):
        fp_key = XCHEM_FPS_COLS_MAP.get(fp)
        y = []
        _x = []
        count = 0
        with gzip.open(path, "rt", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            fp_idx = header.index(fp)
            for line in f:
                if line.strip() == "":
                    continue
                if count == 10000:
                    break
                count += 1
                splits = line.strip().split("\t")
                label = splits[4]
                if label == 'PTE':
                    y.append(1)
                else:
                    y.append(0)
                # y.append(splits[4])
                _x.append([int(_) for _ in splits[fp_idx].split(",")])
        X[fp_key] = np.array(_x)

    return X, y


def _read_xchem_parquet(main_folder, fps):
    X = {}
    y = []

    # Get a list of all parquet files in the main folder
    parquet_files = glob.glob(os.path.join(main_folder, "*.parquet"))

    # df = pd.read_parquet(parquet_files[1])

    # Read all parquet files into a single DataFrame
    df_list = [pd.read_parquet(file) for file in parquet_files]

    df = pd.concat(df_list, ignore_index=True)

    y = df['Label'].apply(lambda label: 1 if label == 'PTE' else 0).tolist()


    for fp in tqdm(fps):
        fp_key = XCHEM_FPS_COLS_MAP.get(fp)

        # Ensure the fp exists in the DataFrame
        if fp not in df.columns:
            raise ValueError(
                f"Fingerprint {fp} not found in the DataFrame columns.")

        X[fp_key] = np.vstack(df[fp].values)

    return X, y
