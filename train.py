import os
import sys
import argparse

from sklearn.model_selection import train_test_split
from typing import List, Tuple, Any
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch

from notebooks import utils
from notebooks.custom_dataset import CustomDataset
from models.pretrained import models, weights


def train(
    model: Any,
    criterion: Any,
    optimizer: Any,
    device: Any,
    trainloader: Any,
    valloader: Any,
):
    model.to(device)
    loss_train_hist: List[float] = []
    loss_eval_hist: List[float] = []
    for epoch in range(args.epochs):
        loss_train = 0.0
        model.train()
        for i, (x, ytrue, ids) in enumerate(trainloader):
            x = x.to(device)
            ytrue = ytrue.to(device)
            optimizer.zero_grad()
            ypred = model(x)
            loss = criterion(ypred.view(-1), ytrue.float())
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        loss_train /= len(trainloader)
        loss_train_hist.append(loss_train)
        model.eval()
        loss_eval = 0.0
        for i, (x, ytrue, ids) in enumerate(valloader):
            x = x.to(device)
            ytrue = ytrue.to(device)
            ypred = model(x)
            loss = criterion(ypred.view(-1), ytrue.float())
            loss_eval += loss.item()
        loss_eval /= len(valloader)
        loss_eval_hist.append(loss_eval)
        print(
            "Epoch: {}, Train Loss: {}, Val Loss: {}".format(
                epoch, loss_train, loss_eval
            )
        )
    model.to("cpu")
    return loss_train_hist, loss_eval_hist


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    root = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        type=str,
        default=os.path.join(root, "train-hdf5/train-image-merge.hdf5"),
    )
    parser.add_argument(
        "--meta_file",
        type=str,
        default=os.path.join(root, "meta/train-metadata-merge.csv"),
    )
    parser.add_argument("--model_name", type=str, default="vit_b_16")
    parser.add_argument(
        "--nsamples",
        type=int,
        nargs="+",
        default=[1233, 1233],
        help="First element for cancer=false=0",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_model", type=bool, default=False)
    parser.add_argument("--plot", type=bool, default=False)
    # TODO add function to clean up all created files except default
    parser.add_argument("--clean_up", type=bool, default=False)
    args = parser.parse_args()
    utils.sample_hdf5_meta(
        data_file=args.data_file,
        meta_file=args.meta_file,
        n_sample_1=args.nsamples[1],
        n_sample_0=args.nsamples[0],
    )
    weight = weights[args.model_name]
    args.weight_name = weight.name
    model_f = models[args.model_name]
    model = model_f(weights=weight)
    args.model_class_name = model.__class__.__name__
    model = utils.get_adjusted_model(model)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    transform = weight.transforms()
    dataset = CustomDataset(
        data_file=args.data_file.replace(
            ".hdf5", f"-{args.nsamples[0] + args.nsamples[1]}.hdf5"
        ),
        meta_file=args.meta_file.replace(
            ".csv", f"-{args.nsamples[0] + args.nsamples[1]}.csv"
        ),
        transform=transform,
    )
    if not utils.data_matches_model_spec(dataset[0], weight):
        raise ValueError("Data does not match model specification")
    trainset, valset = train_test_split(dataset, test_size=0.2)
    trainset: List[Tuple[torch.Tensor, np.int64, str]] = trainset
    valset: List[Tuple[torch.Tensor, np.int64, str]] = valset
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device(args.device)
    train_hist, eval_hist = train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        trainloader=trainloader,
        valloader=valloader,
    )
    if args.save_model:
        utils.save_model(model, args)
    score = utils.get_pAUC(model, valloader)
    print("Partial AUC: ", score)
    if args.plot:
        utils.save_plot(train_hist, eval_hist, args, score)
