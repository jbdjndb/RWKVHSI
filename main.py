import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.utils.data as data
import numpy as np
import seaborn as sns
import os

from utils import (
    metrics,
    convert_to_color_,
    convert_from_color_,
    sample_gt,
    build_dataset,
    show_results,
    compute_imf_weights,
    get_device,
)
from datasets import get_dataset, HyperX, open_file, DATASETS_CONFIG
from models import get_model, train, test, save_model
import argparse

viz = None

dataset_names = [v["name"] if "name" in v.keys() else k for k, v in DATASETS_CONFIG.items()]

parser = argparse.ArgumentParser(description="Run deep learning experiments on various hyperspectral datasets")
parser.add_argument("--dataset", type=str, default="IndianPines", choices=dataset_names)
parser.add_argument("--model", type=str, default="RWKVHSI")
parser.add_argument("--folder", type=str, default="./Datasets/")
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--runs", type=int, default=10)
parser.add_argument("--restore", type=str, default=None)

group_dataset = parser.add_argument_group("Dataset")
group_dataset.add_argument("--training_sample", type=float, default=0.1)
group_dataset.add_argument("--sampling_mode", type=str, default="random")
group_dataset.add_argument("--train_set", type=str, default=None)
group_dataset.add_argument("--test_set", type=str, default=None)

group_train = parser.add_argument_group("Training")
group_train.add_argument("--epoch", type=int)
group_train.add_argument("--patch_size", type=int)
group_train.add_argument("--lr", type=float)
group_train.add_argument("--class_balancing", action="store_true")
group_train.add_argument("--batch_size", type=int)
group_train.add_argument("--test_stride", type=int, default=1)

group_da = parser.add_argument_group("Data augmentation")
group_da.add_argument("--flip_augmentation", action="store_true")
group_da.add_argument("--radiation_augmentation", action="store_true")
group_da.add_argument("--mixture_augmentation", action="store_true")

parser.add_argument("--with_exploration", action="store_true")
parser.add_argument("--download", type=str, default=None, nargs="+", choices=dataset_names)

args = parser.parse_args()

CUDA_DEVICE = get_device(args.cuda)
SAMPLE_PERCENTAGE = args.training_sample
FLIP_AUGMENTATION = args.flip_augmentation
RADIATION_AUGMENTATION = args.radiation_augmentation
MIXTURE_AUGMENTATION = args.mixture_augmentation
DATASET = args.dataset
MODEL = args.model
N_RUNS = args.runs
PATCH_SIZE = args.patch_size
DATAVIZ = args.with_exploration
FOLDER = args.folder
EPOCH = args.epoch
SAMPLING_MODE = args.sampling_mode
CHECKPOINT = args.restore
LEARNING_RATE = args.lr
CLASS_BALANCING = args.class_balancing
TRAIN_GT = args.train_set
TEST_GT = args.test_set
TEST_STRIDE = args.test_stride

if args.download is not None:
    for dataset in args.download:
        get_dataset(dataset, target_folder=FOLDER)
    quit()

hyperparams = vars(args)
img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET, FOLDER)
N_CLASSES = len(LABEL_VALUES)
N_BANDS = img.shape[-1]

if palette is None:
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))
invert_palette = {v: k for k, v in palette.items()}

def convert_to_color(x):
    return convert_to_color_(x, palette=palette)
def convert_from_color(x):
    return convert_from_color_(x, palette=invert_palette)

hyperparams.update({
    "n_classes": N_CLASSES,
    "n_bands": N_BANDS,
    "ignored_labels": IGNORED_LABELS,
    "device": CUDA_DEVICE,
})
hyperparams = {k: v for k, v in hyperparams.items() if v is not None}

results = []

for run in range(N_RUNS):
    if TRAIN_GT and TEST_GT:
        train_gt = open_file(TRAIN_GT)
        test_gt = open_file(TEST_GT)
    elif TRAIN_GT:
        train_gt = open_file(TRAIN_GT)
        test_gt = np.copy(gt)
        w, h = test_gt.shape
        test_gt[(train_gt > 0)[:w, :h]] = 0
    elif TEST_GT:
        test_gt = open_file(TEST_GT)
    else:
        train_gt, test_gt = sample_gt(gt, SAMPLE_PERCENTAGE, mode=SAMPLING_MODE)

    print(f"{np.count_nonzero(train_gt)} train samples, {np.count_nonzero(test_gt)} test samples")
    print(f"Run {run+1}/{N_RUNS} with {MODEL}")

    if CLASS_BALANCING:
        weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
        hyperparams["weights"] = torch.from_numpy(weights)

    model, optimizer, loss, hyperparams = get_model(MODEL, **hyperparams)
    train_dataset = HyperX(img, train_gt, **hyperparams)
    train_loader = data.DataLoader(train_dataset, batch_size=hyperparams["batch_size"], shuffle=True)

    if CHECKPOINT is not None:
        pre_model = torch.load(CHECKPOINT, map_location='cpu')
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in pre_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    try:
        train(
            model, optimizer, loss, train_loader, hyperparams["epoch"],
            scheduler=hyperparams["scheduler"], device=hyperparams["device"],
            supervision=hyperparams["supervision"], display=viz,
        )
    except KeyboardInterrupt:
        pass

    probabilities = test(model, img, hyperparams)
    prediction = np.argmax(probabilities, axis=-1)

    run_results = metrics(prediction, test_gt, ignored_labels=hyperparams["ignored_labels"], n_classes=N_CLASSES)

    mask = np.zeros(gt.shape, dtype="bool")
    for l in IGNORED_LABELS:
        mask[gt == l] = True
    prediction[mask] = 0

    txt_dir = f"./cls_result/{DATASET}/{MODEL}/"
    os.makedirs(txt_dir, exist_ok=True)
    txt_path = txt_dir + f"{run+1}.txt"
    results.append(run_results)
    show_results(run_results, None, label_values=LABEL_VALUES, txt_path=txt_path)

if N_RUNS > 1:
    txt_dir = f"./cls_result/{DATASET}/{MODEL}/"
    os.makedirs(txt_dir, exist_ok=True)
    txt_path = txt_dir + f"{N_RUNS}_all.txt"
    show_results(results, None, label_values=LABEL_VALUES, agregated=True, txt_path=txt_path)