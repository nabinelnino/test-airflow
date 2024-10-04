import numpy as np
import pandas as pd


def early_enrichment(y, y_pred, _top_n: int = 24):
    y_pred = np.atleast_1d(y_pred)
    y = np.atleast_1d(y)
    
    # Sort based on y_pred descending and select top _top_n rows
    _tmp = np.vstack((y, y_pred)).T[y_pred.argsort()[::-1]][:_top_n, :]
    
    # Filter rows where y_pred > 0.5 and remove any NaNs
    _tmp = _tmp[np.where((_tmp[:, 1] > 0.5) & (~np.isnan(_tmp[:, 0])))[0]].copy()
    
    # Check if _tmp is not empty to avoid division by zero
    if len(_tmp) > 0:
        return np.sum(_tmp[:, 0]) / len(_tmp)
    else:
        return 0.0 

def diverse_early_enrichment(y, y_pred, groups, top_n_per_group: int = 10):
    df = pd.DataFrame({"CLUSTER_ID": groups, "pred": y_pred, "real": y})
    df_groups = df.groupby("CLUSTER_ID")

    _vals = []
    for group, idx in df_groups.groups.items():
        _tmp = df.iloc[idx].copy()
        if sum(df.iloc[idx]["pred"] > 0.5) == 0:
            continue
        _tmp = _tmp[_tmp["pred"] > 0.5].copy()
        _tmp = np.vstack((_tmp["real"].to_numpy(), _tmp["pred"].to_numpy())).T[
                   _tmp["pred"].to_numpy().argsort()[::-1]][:top_n_per_group, :]
        _val = np.sum(_tmp[:, 0]) / len(_tmp)
        _vals.append(_val)

    return np.mean(_vals)
