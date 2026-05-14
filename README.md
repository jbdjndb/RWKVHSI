# RWKVHSI: Hyperspectral Image Classification with RWKV-Based Spectral-Spatial Modeling

This repository contains a PyTorch implementation of **RWKVHSI**, a lightweight RWKV-based model for hyperspectral image classification. The project includes the model code, experiment entry point, dataset loaders, metric utilities, and a reproducible pip environment file.

RWKVHSI combines:

- RWKV-style `TimeMix` and `ChannelMix` blocks for sequence-like feature interaction.
- Separate spectral and spatial streams.
- Adaptive gated fusion for spectral-spatial feature integration.
- Patch-based training and sliding-window inference on full hyperspectral scenes.

## Repository Structure

```text
.
|-- README.md             # Project usage guide
|-- environment.txt       # Reproduction environment based on the paper settings
|-- main.py               # Experiment entry point
|-- models.py             # Model factory, training loop, inference loop, checkpoint saving
|-- datasets.py           # Dataset loaders and HyperX patch dataset
|-- utils.py              # Sampling, metrics, result saving, and helper functions
|-- model/
|   |-- RWKVHSI.py        # RWKVHSI architecture
|   `-- __init__.py~      # Backup file
|-- Datasets/             # User-created dataset folder, not included in GitHub
|   |-- IndianPines/
|   |-- Houston/
|   `-- whulk/
`-- __pycache__/          # Python cache files
```

Large hyperspectral `.mat` files are not included in this repository because they are too large for normal GitHub hosting. Download the public datasets separately and place them under `Datasets/` as described below. Files ending in `~` and files under `__pycache__/` are backup/cache artifacts and are not required for normal training or evaluation.

## Environment

The recommended reproduction environment is recorded in `environment.txt`. It follows the experimental platform described in the paper:

- OS: Ubuntu 22.04
- CPU: Intel Xeon Silver 4310
- GPU: 2 x NVIDIA RTX 4090
- CUDA: 11.8
- PyTorch: 2.1.1
- Recommended Python: 3.10

Create and install the environment:

```bash
conda create -n rwkvhsi python=3.10
conda activate rwkvhsi
pip install -r environment.txt
```

`environment.txt` installs the CUDA 11.8 PyTorch wheel through the PyTorch extra index. For CPU-only usage, install a CPU build of PyTorch separately, then keep the remaining package versions unchanged.

The required third-party Python packages are:

- `torch`
- `numpy`
- `scipy`
- `scikit-learn`
- `spectral`
- `seaborn`
- `matplotlib`
- `tqdm`
- `einops`
- `joblib`

## Dataset Preparation

Datasets are not distributed with this repository. By default, `main.py` reads datasets from `./Datasets/`, so create this folder manually after downloading the public data.

The expected directory structure is:

```text
Datasets/
|-- IndianPines/
|   |-- Indian_pines_corrected.mat
|   `-- Indian_pines_gt.mat
|-- whulk/
|   |-- WHU_Hi_LongKou.mat
|   `-- WHU_Hi_LongKou_gt.mat
`-- Houston/
    |-- Houston.mat
    `-- Houston_gt.mat
```

Runnable dataset choices with the current loaders are:

- `--dataset IndianPines`
- `--dataset whulk`
- `--dataset Houston`

### Download Sources

- **Indian Pines**: The code can download this dataset automatically:

```bash
python main.py --download IndianPines
```

  You can also download the public `.mat` files from the links documented by [TensorLy Indian Pines](https://tensorly.org/stable/modules/generated/tensorly.datasets.load_indian_pines.html). The required filenames are `Indian_pines_corrected.mat` and `Indian_pines_gt.mat`.

- **WHU-Hi-LongKou**: Download the Matlab data format from the official [WHU-Hi resource page](https://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm), then place `WHU_Hi_LongKou.mat` and `WHU_Hi_LongKou_gt.mat` under `Datasets/whulk/`.

- **Houston2013**: Request or download the public Houston dataset from the [Houston 2013 dataset page](https://hyperspectral.ee.uh.edu/?page_id=459). The current `Houston` loader expects `Datasets/Houston/Houston.mat` with variable name `data`, and `Datasets/Houston/Houston_gt.mat` with variable name `groundT`. If your downloaded files use different names, rename the files, adjust the `.mat` variable names, or update the `Houston` branch in `datasets.py`.

Note: although `datasets.py` has a `DATASETS_CONFIG` entry for `whulk`, its URL list is not the real WHU-Hi-LongKou download source. Download WHU-Hi-LongKou manually from the official WHU-Hi page. The `Houston` entry has `download=False`, so it also must be prepared manually.

## Quick Start

Run one experiment on Indian Pines:

```bash
python main.py --dataset IndianPines --model RWKVHSI --runs 1
```

Run one experiment on WHU-Hi-LongKou:

```bash
python main.py --dataset whulk --model RWKVHSI --runs 1
```

Run one experiment on Houston2013 after preparing the files:

```bash
python main.py --dataset Houston --model RWKVHSI --runs 1
```

Run a paper-like Indian Pines setting with 10% training samples and augmentation:

```bash
python main.py --dataset IndianPines --model RWKVHSI --runs 10 --training_sample 0.1 --epoch 100 --batch_size 32 --flip_augmentation --radiation_augmentation
```

Run a paper-like WHU-Hi-LongKou setting with 1% training samples:

```bash
python main.py --dataset whulk --model RWKVHSI --runs 10 --training_sample 0.01 --epoch 100 --batch_size 32 --flip_augmentation --radiation_augmentation
```

Run on CPU:

```bash
python main.py --dataset IndianPines --model RWKVHSI --runs 1 --cuda -1
```

Restore a checkpoint before training:

```bash
python main.py --dataset IndianPines --model RWKVHSI --runs 1 --restore checkpoints/rwkvhsi/IndianPines/example.pth
```

## Main Arguments

| Argument | Description | Default |
| --- | --- | --- |
| `--dataset` | Dataset name. Prepared local choices include `IndianPines`, `whulk`, and `Houston`. | `IndianPines` |
| `--model` | Model name registered in `models.py`. | `RWKVHSI` |
| `--folder` | Root folder containing dataset subdirectories. | `./Datasets/` |
| `--cuda` | CUDA device index. Use `-1` for CPU. | `0` |
| `--runs` | Number of repeated train/test runs. | `10` |
| `--restore` | Path to a checkpoint state dict to load before training. | `None` |
| `--training_sample` | Fraction or count of labeled pixels used for training. | `0.1` |
| `--sampling_mode` | Sampling strategy. Implemented modes include `random`, `fixed`, and `disjoint`. | `random` |
| `--train_set` | Optional external training ground-truth file. | `None` |
| `--test_set` | Optional external testing ground-truth file. | `None` |
| `--epoch` | Number of training epochs. | `100` in `models.py` |
| `--patch_size` | Spatial patch size. | `9` in `models.py` |
| `--batch_size` | Training batch size. | `32` in `models.py` |
| `--lr` | Parsed by `main.py`, but see the note below. | `None` |
| `--class_balancing` | Enable inverse median frequency class weights. | disabled |
| `--test_stride` | Sliding-window stride during inference. | `1` |
| `--flip_augmentation` | Enable random flip augmentation. | disabled |
| `--radiation_augmentation` | Enable radiation noise augmentation. | disabled |
| `--mixture_augmentation` | Enable mixture noise augmentation. | disabled |
| `--download` | Download supported public datasets listed in `datasets.py`. For this project, use it for `IndianPines`; prepare `whulk` and `Houston` manually. | `None` |

## Outputs

Experiment metrics are saved under:

```text
cls_result/<DATASET>/<MODEL>/
```

For example:

```text
cls_result/IndianPines/RWKVHSI/1.txt
```

For multiple runs, each run is saved as `1.txt`, `2.txt`, and so on. Aggregated results are saved as:

```text
cls_result/<DATASET>/<MODEL>/<runs>_all.txt
```

During long training runs, checkpoints may be saved under:

```text
checkpoints/<model>/<dataset>/
```

## Model And Training Defaults

The current `RWKVHSI` setup in `models.py` uses:

- Patch size: `9`
- Batch size: `32`
- Epochs: `100`
- Optimizer: Adam
- Learning rate in code: `0.001`
- Weight decay: `1e-4`
- Scheduler: `StepLR` inside the training loop, with step size `10` and gamma `0.7`

The paper environment section reports 100 epochs, Adam, initial learning rate `0.01`, batch size `32`, random flipping, and radiation noise augmentation. The current code parses `--lr`, but `get_model()` reads the internal `learning_rate` key and defaults to `0.001`, so changing the learning rate from the command line requires adjusting the argument mapping or editing `models.py`.

## Notes

- Keep `--batch_size` greater than `1` unless you have checked the tensor shapes carefully. The model calls `squeeze()` in `RWKVHSI.forward()`, which can be fragile for singleton batches.
- `environment.txt` targets the paper-level CUDA reproduction environment, not necessarily the current local Python installation.
- `python main.py --help` requires all import-time dependencies to be installed, including `spectral`.
- No benchmark numbers are included here because results depend on data splits, random seeds, environment details, and local code changes.
