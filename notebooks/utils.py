from sklearn.metrics import roc_curve, auc, roc_auc_score
import pandas.api.types
import numpy as np
import pandas as pd
import h5py

TRAIN_META = "../meta/train-metadata.csv"
DATA_PATH = "../train-hdf5/train-image.hdf5"
ID_NAME = "isic_id"


def get_ground_truth_df(df_path: str) -> pd.DataFrame:
    df = pd.read_csv(df_path)
    assert "target" in df.columns
    assert "isic_id" in df.columns
    return pd.DataFrame(df[["isic_id", "target"]].copy())


def sample_hdf5_meta(
    out_data_file: str,
    out_meta_file: str,
    n_sample_1: int,
    n_sample_0: int,
    meta_path: str = TRAIN_META,
    data_path: str = DATA_PATH,
    id_name: str = ID_NAME,
) -> None:
    """Creates two new files, once with the sampled HDF5 data and the other with the corresponding metadata."""
    df_meta = pd.read_csv(meta_path)
    len1 = len(df_meta[df_meta["target"] == 1])
    assert n_sample_1 <= len1, f"n_sample_1={n_sample_1} cannot be greater than {len1}"
    df_1 = df_meta[df_meta["target"] == 1].sample(n=n_sample_1)
    df_0 = df_meta[df_meta["target"] != 1].sample(n=n_sample_0)
    df_sample = pd.concat([df_1, df_0])
    ids = df_sample[id_name].values.tolist()
    with h5py.File(data_path, 'r') as f:
        with h5py.File(out_data_file, "w") as f_subset:
            for id_ in ids:
                data = f[id_][()]
                f_subset.create_dataset(id_, data=data)
    df_sample.to_csv(out_meta_file, index=False)
    return None


def score(solution: pd.DataFrame,
          submission: pd.DataFrame,
          row_id_column_name: str,
          min_tpr: float = 0.80) -> float:
    '''
    2024 ISIC Challenge metric: pAUC

    Given a solution file and submission file, this function returns the
    the partial area under the receiver operating characteristic (pAUC) 
    above a given true positive rate (TPR) = 0.80.
    https://en.wikipedia.org/wiki/Partial_Area_Under_the_ROC_Curve.

    (c) 2024 Nicholas R Kurtansky, MSKCC

    Args:
        solution: ground truth pd.DataFrame of 1s and 0s
        submission: solution dataframe of predictions of scores ranging [0, 1]

    Returns:
        Float value range [0, max_fpr]
    '''

    del solution[row_id_column_name]
    del submission[row_id_column_name]

    # check submission is numeric
    if not pandas.api.types.is_numeric_dtype(submission.values):
        raise ValueError(
            "Expected numeric values in submission, got: %r" % submission.values)

    # rescale the target. set 0s to 1s and 1s to 0s (since sklearn only has max_fpr)
    # v_gt = abs(np.asarray(solution.values)-1)
    v_gt = solution.values.reshape(-1)

    # flip the submissions to their compliments
    # v_pred = -1.0*np.asarray(submission.values)
    v_pred = submission.values.reshape(-1)

    max_fpr = abs(1-min_tpr)

    # using sklearn.metric functions: (1) roc_curve and (2) auc
    fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)

    # Add a single point at max_fpr by linear interpolation
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    partial_auc = auc(fpr, tpr)

#     # Equivalent code that uses sklearn's roc_auc_score
#     v_gt = abs(np.asarray(solution.values)-1)
#     v_pred = np.array([1.0 - x for x in submission.values])
#     max_fpr = abs(1-min_tpr)
#     partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
#     # change scale from [0.5, 1.0] to [0.5 * max_fpr**2, max_fpr]
#     # https://math.stackexchange.com/questions/914823/shift-numbers-into-a-different-range
#     partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)

    return (partial_auc)
