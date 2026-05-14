# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn import init
from model.RWKVHSI import RWKVHSI
import os
import datetime
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.decomposition import PCA
from utils import grouper, sliding_window, count_sliding_window, camel_to_snake, padding_image
from torch.optim.lr_scheduler import StepLR

def get_model(name, **kwargs):
    device = kwargs.setdefault("device", torch.device("cpu"))
    n_classes = kwargs["n_classes"]
    n_bands = kwargs["n_bands"]
    weights = torch.ones(n_classes)
    weights[torch.LongTensor(kwargs["ignored_labels"])] = 0.0
    weights = weights.to(device)
    weights = kwargs.setdefault("weights", weights)

    if name == "RWKVHSI":
        kwargs.setdefault("patch_size", 9)
        center_pixel = True
        model = RWKVHSI(
            in_channels=n_bands, num_classes=n_classes,
        )
        lr = kwargs.setdefault("learning_rate", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 32)
    else:
        raise KeyError("{} model is unknown.".format(name))

    model = model.to(device)
    epoch = kwargs.setdefault("epoch", 100)
    kwargs.setdefault(
        "scheduler",
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=epoch // 4
        ),
    )
    kwargs.setdefault("supervision", "full")
    kwargs["center_pixel"] = center_pixel
    return model, optimizer, criterion, kwargs


def train(
    net,
    optimizer,
    criterion,
    data_loader,
    epoch,
    scheduler=None,
    display_iter=100,
    device=torch.device("cpu"),
    display=None,
    val_loader=None,
    supervision="full",
):
    net.to(device)

    if hasattr(criterion,"weight") and criterion.weight is not None:
        criterion.weight = criterion.weight.to(device)

    save_epoch = epoch // 50 if epoch > 50 else 0
    losses = np.zeros(10000000)
    mean_losses = np.zeros(10000000)
    iter_ = 1
    gamma = 0.7
    scheduler = StepLR(optimizer, step_size=10, gamma=gamma)

    use_cuda_timing = torch.cuda.is_available() and device.type == 'cuda'
    if use_cuda_timing:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    for e in tqdm(range(1, epoch + 1), desc="Training the network"):
        net.train()
        avg_loss = 0.0

        for batch_idx, (data, target) in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            output = net(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100) : iter_ + 1])
            iter_ += 1
            del (data, target, loss, output)

        avg_loss /= len(data_loader)
        if val_loader is not None:
            val_acc = val(net, val_loader, device=device)
            metric = -val_acc
        else:
            metric = avg_loss

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        elif scheduler is not None:
            scheduler.step()

        if save_epoch > 0 and e % save_epoch == 0:
            save_model(
                net,
                camel_to_snake(str(net.__class__.__name__)),
                data_loader.dataset.name,
                epoch=e,
                metric=abs(metric),
            )

    if use_cuda_timing:
        end.record()
        torch.cuda.synchronize()
        print("The training time is:***********************", start.elapsed_time(end)/1000)
    else:
        print("The training time is:*********************** (CPU mode, no CUDA timing)")


def save_model(model, model_name, dataset_name, **kwargs):
    model_dir = "./checkpoints/" + model_name + "/" + dataset_name + "/"
    time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if isinstance(model, torch.nn.Module):
        filename = time_str + "_epoch{epoch}_{metric:.2f}".format(**kwargs)
        torch.save(model.state_dict(), model_dir + filename + ".pth")
    else:
        joblib.dump(model, model_dir + time_str + ".pkl")


def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX.astype("float32")


def test(net, img, hyperparams):
    net.eval()
    patch_size = hyperparams["patch_size"]
    center_pixel = hyperparams["center_pixel"]
    batch_size, device = hyperparams["batch_size"], hyperparams["device"]
    n_classes = hyperparams["n_classes"]

    use_cuda_timing = torch.cuda.is_available() and device.type == 'cuda'
    if use_cuda_timing:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    kwargs = {
        "step": hyperparams["test_stride"],
        "window_size": (patch_size, patch_size),
    }
    probs = np.zeros(img.shape[:2] + (n_classes,))
    iterations = count_sliding_window(img, **kwargs) // batch_size

    for batch in tqdm(
        grouper(batch_size, sliding_window(img, **kwargs)),
        total=iterations,
        desc="Inference on the image",
    ):
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)
            else:
                data = [b[0] for b in batch]
                data = np.copy(data)
                data = data.transpose(0, 3, 1, 2)
                data = torch.from_numpy(data)
                data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]
            data = data.to(device)
            # output = net(data)
            output = net(data.unsqueeze(1))  # 增加通道维度
            if isinstance(output, tuple):
                output = output[0]
            output = output.to("cpu").numpy()

            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x:x+w, y:y+h] += out

    if use_cuda_timing:
        end.record()
        torch.cuda.synchronize()
        print("The testing time is:***********************", start.elapsed_time(end)/1000)
    else:
        print("The testing time is:*********************** (CPU mode, no CUDA timing)")

    return probs


def val(net, data_loader, device="cpu"):
    accuracy, total = 0.0, 0.0
    ignored_labels = data_loader.dataset.ignored_labels
    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            data, target = data.to(device), target.to(device)
            output = net(data)
            _, output = torch.max(output, dim=1)
            for out, pred in zip(output.view(-1), target.view(-1)):
                if out.item() in ignored_labels:
                    continue
                accuracy += out.item() == pred.item()
                total += 1
    return accuracy / total if total > 0 else 0