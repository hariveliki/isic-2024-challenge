import io
from PIL import Image
from copy import deepcopy
import os

from sklearn.metrics import roc_curve, auc, roc_auc_score
from typing import Any, List, Tuple
import matplotlib.pyplot as plt
import pandas.api.types
import numpy as np
import pandas as pd
import h5py
import torch.nn as nn
import torch


def get_ground_truth_df(df_path: str) -> pd.DataFrame:
    df = pd.read_csv(df_path)
    assert "target" in df.columns
    assert "isic_id" in df.columns
    return pd.DataFrame(df[["isic_id", "target"]].copy())


def sample_hdf5_meta(
    data_file: str,
    meta_file: str,
    n_sample_1: int,
    n_sample_0: int,
    id_name: str = "isic_id",
) -> None:
    """Creates two new files, once with the sampled HDF5 data and the other with the corresponding metadata."""
    n_samples = n_sample_1 + n_sample_0
    out_data_file = data_file.replace(".hdf5", f"-{n_samples}.hdf5")
    out_meta_file = meta_file.replace(".csv", f"-{n_samples}.csv")
    if files_exist([out_data_file, out_meta_file]):
        return None
    df_meta = pd.read_csv(meta_file)
    len1 = len(df_meta[df_meta["target"] == 1])
    assert n_sample_1 <= len1, f"n_sample_1={n_sample_1} cannot be greater than {len1}"
    df_1 = df_meta[df_meta["target"] == 1].sample(n=n_sample_1)
    df_0 = df_meta[df_meta["target"] != 1].sample(n=n_sample_0)
    df_sample = pd.concat([df_1, df_0])
    ids = df_sample[id_name].values.tolist()
    with h5py.File(data_file, "r") as f:
        with h5py.File(out_data_file, "w") as f_subset:
            for id_ in ids:
                byte_data = f[id_][()]
                image = Image.open(io.BytesIO(byte_data))

                with io.BytesIO() as buffer:
                    image.save(buffer, format="JPEG")
                    image_bytes = buffer.getvalue()

                f_subset.create_dataset(id_, data=np.void(image_bytes))
    df_sample.to_csv(out_meta_file, index=False)
    return None


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    min_tpr: float = 0.80,
) -> float:
    """
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
    """

    del solution[row_id_column_name]
    del submission[row_id_column_name]

    # check submission is numeric
    if not pandas.api.types.is_numeric_dtype(submission.values):
        raise ValueError(
            "Expected numeric values in submission, got: %r" % submission.values
        )

    # rescale the target. set 0s to 1s and 1s to 0s (since sklearn only has max_fpr)
    # v_gt = abs(np.asarray(solution.values)-1)
    v_gt = solution.values.reshape(-1)

    # flip the submissions to their compliments
    # v_pred = -1.0*np.asarray(submission.values)
    v_pred = submission.values.reshape(-1)

    max_fpr = abs(1 - min_tpr)

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

    return partial_auc


def get_adjusted_model(model: Any) -> Any:
    """Adjust the fully connected layer."""
    if model.__class__.__name__ == "ResNet":
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1), nn.Sigmoid())
    elif model.__class__.__name__ == "VisionTransformer":
        model.heads = nn.Sequential(
            nn.Linear(model.heads.head.in_features, 1), nn.Sigmoid()
        )
    else:
        raise NotImplementedError
    return model


def files_exist(files: List[str]) -> bool:
    return all([os.path.exists(f) for f in files])


def data_matches_model_spec(
    dataset: Tuple[torch.Tensor, np.int64, str], weight: Any
) -> bool:
    # TODO make generic
    """Check if shape of tensor matches the model specification."""
    (C, H, W) = dataset[0].shape
    model_crop_size: int = weight.transforms().crop_size[0]
    return H == W == model_crop_size


def save_model(model: Any, args: Any) -> None:
    samples_name = "_".join([str(n) for n in args.nsamples])
    model_weight_name = f"{args.model_class_name}_{args.weight_name}"
    os.makedirs(os.path.join("models", model_weight_name), exist_ok=True)
    model_file = os.path.join("models", model_weight_name, f"{samples_name}_model.pth")
    best_model = deepcopy(model.state_dict())
    torch.save(best_model, model_file)
    return None


def get_pAUC(model: Any, dataloader: torch.utils.data.DataLoader) -> None:
    model.eval()
    solution = []
    submission = []
    indices = []
    with torch.no_grad():
        for i, (x, ytrue, ids) in enumerate(dataloader):
            ypred = model(x)
            ypred = ypred.flatten()
            solution.extend(ytrue.numpy())
            submission.extend(ypred.numpy())
            indices.extend(ids)
    submission = pd.DataFrame(submission, columns=["yhat"])
    submission["isic_id"] = indices
    solution = pd.DataFrame(solution, columns=["ytrue"])
    solution["isic_id"] = indices
    return score(solution, submission, "isic_id")


def save_plot(
    train_hist: List[float], eval_hist: List[float], args: Any, score: float
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("pAUC Score: {}".format(score))
    ax1.plot(train_hist, label="Train Loss")
    ax1.set_title("Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax2.plot(eval_hist, label="Val Loss")
    ax2.set_title("Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    plt.tight_layout()
    plt.show()
    samples_name = "_".join([str(n) for n in args.nsamples])
    model_weight_name = f"{args.model_class_name}_{args.weight_name}"
    os.makedirs(os.path.join("plots", model_weight_name), exist_ok=True)
    plot_file = os.path.join("plots", model_weight_name, f"{samples_name}_loss.png")
    fig.savefig(plot_file)
    return None


def get_ids_for_target(target: int, meta_dir: str) -> List[str]:
    df = pd.read_csv(meta_dir)
    return df[df["target"] == target]["isic_id"].values.tolist()


def save_imgs_by_target(
    target: int, meta_dir: str, data_dir: str, out_dir: str
) -> None:
    ids = get_ids_for_target(target, meta_dir)
    for file in os.listdir(data_dir):
        if os.path.splitext(file)[0] in ids:
            os.system(
                f"cp {os.path.join(data_dir, file)} {os.path.join(out_dir, file)}"
            )
    return None


def merge_hdf5_files(file1: str, file2: str, output_file: str) -> None:
    with h5py.File(file1, "r") as f1, h5py.File(file2, "r") as f2, h5py.File(
        output_file, "w"
    ) as f_out:

        for key in f1.keys():
            f1.copy(key, f_out)

        for key in f2.keys():
            f2.copy(key, f_out)


def save_imgs_to_hdf5(img_dir: List[str], out_file: str):
    files = os.listdir(img_dir)
    image_paths = [os.path.join(img_dir, file) for file in files]
    with h5py.File(out_file, "w") as hdf5_file:
        for i, image_path in enumerate(image_paths):
            dir_, filename = os.path.split(image_path)
            filename, filetype = os.path.splitext(filename)
            image = Image.open(image_path)

            with io.BytesIO() as buffer:
                image.save(buffer, format=image.format)
                image_bytes = buffer.getvalue()

            hdf5_file.create_dataset(filename, data=np.void(image_bytes))


def convert_png_to_jpg(in_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for file_name in os.listdir(in_dir):
        if not file_name.endswith(".png"):
            continue
        with Image.open(f"variations/{file_name}") as img:
            rgb_img = img.convert("RGB")
            new_path = os.path.join(out_dir, file_name.replace(".png", ".jpg"))
            rgb_img.save(new_path, "JPEG")


def create_meta_for_variations(metapath: str, hdf5_path: str, outfile: str) -> None:
    with h5py.File(hdf5_path, "r") as f:
        keys = list(f.keys())
        keys_set = set(keys)

    metadf = pd.read_csv(metapath)
    ids = metadf["isic_id"].values
    ids_set = set(ids)
    ids_diff = keys_set.difference(ids_set)
    new_df = pd.DataFrame({"isic_id": list(ids_diff), "target": 1})
    concat_df = pd.concat([metadf, new_df], ignore_index=True)
    concat_df.to_csv(outfile, index=False)
    return None
