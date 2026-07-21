# WhiteCon

WhiteCon: Semi-Supervised Domain Adaptation Regression Through Whitening Transform and Dual Consistency

## Requirements

Python 3.12.7 with CUDA 11.8:

```bash
pip install -r requirements.txt
```

Experiments were run on a single NVIDIA GeForce RTX 4090 (Intel Core i9-14900KF, 64 GB RAM).

## Dataset

- BIWI can be downloaded from [here (faces-0)](https://www.kaggle.com/datasets/kmader/biwi-kinect-head-pose-database/data?select=db_annotations).
- QMUL can be downloaded from [here](http://www.eecs.qmul.ac.uk/~sgg/QMUL_FaceDataset/QMULFaceDataset.zip).
- MPI3D can be downloaded from [here](https://drive.google.com/drive/folders/12iHhUdVl-CyywQ8fkiM3UssF4ci_S0ZZ).

### Expected directory layout

This repository already ships the split files (`*_source.txt`, `*_labeled_*.txt`, `*_unlabeled.txt`,
`*_valid.txt`, `*_test.txt`) and the BIWI bounding boxes (`data/biwi/hpdb/Biwi_plus.npz`).
Only the **images** have to be downloaded and placed as shown below. All paths inside the split
files are resolved relative to the repository root, so training **must be launched from the
repository root**.

```
WhiteCon/
├── data/
│   ├── biwi/
│   │   ├── F_source.txt, F_labeled_{5,10,20,30,full}.txt, F_unlabeled.txt, F_valid.txt, F_test.txt
│   │   ├── M_source.txt, M_labeled_{5,10,20,30,full}.txt, M_unlabeled.txt, M_valid.txt, M_test.txt
│   │   └── hpdb/                      # <- unzip BIWI `faces_0` here
│   │       ├── Biwi_plus.npz          #    (already provided by this repository)
│   │       ├── 01/
│   │       │   ├── frame_00003_rgb.png
│   │       │   ├── frame_00003_pose.txt
│   │       │   └── ...
│   │       ├── 02/ ... 14/            #    subjects 01–14; the F/M domain split is fixed by the .txt files
│   ├── qmul/
│   │   ├── F_source.txt, F_labeled_*.txt, F_unlabeled.txt, F_valid.txt, F_test.txt
│   │   ├── M_source.txt, M_labeled_*.txt, M_unlabeled.txt, M_valid.txt, M_test.txt
│   │   └── images/                    # <- all QMUL images, flattened into a single folder
│   │       ├── Set1_Greyscale_AndreeaV_060_000.jpg
│   │       ├── Set2_Colour_...jpg
│   │       └── ...
│   └── mpi3d/
│       ├── real_source.txt, real_labeled_*.txt, real_unlabeled.txt, real_valid.txt, real_test.txt
│       ├── realistic_*.txt, toy_*.txt
│       ├── real/                      # <- MPI3D images, one folder per domain
│       │   ├── 1000010.jpg
│       │   └── ...
│       ├── realistic/
│       └── toy/
├── dataloaders/
├── method/WhiteCon.py                 # Trainer (training loop, losses, evaluation)
├── models/                            # ResNet backbone with whitening transform + regressor
├── utils/args.py                      # all command-line arguments
└── main_proposed.py                   # entry point
```

Notes on the split-file formats (useful if you want to regenerate them):

- **BIWI** — `<image path> <pose path>`, both relative to `data/biwi/hpdb/`.
  Labels (yaw, pitch, roll) are derived from the rotation matrix in the pose file.
- **QMUL** — one full path per line, e.g. `data/qmul/images/Set1_Greyscale_AndreeaV_060_000.jpg`.
  Labels (tilt, pan) are parsed from the last two fields of the file name.
- **MPI3D** — `<domain>/<image>.jpg <label1> <label2>`, relative to `data/mpi3d/`.

## Usage

All experiments are launched through `main_proposed.py` from the repository root.

```bash
# BIWI (domains F, M), F -> M, 5% labeled target data
python main_proposed.py --dataset biwi --source F --target M --label-target-per 5 \
                        --result-name biwi_F2M_5

# MPI3D uses different domain names: real, realistic, toy (RL, RC, T in the paper)
python main_proposed.py --dataset mpi3d --source real --target realistic \
                        --label-target-per 5 --result-name mpi3d_RL2RC_5
```
`--label-target-per` selects how much of the target labeled pool
(`data/<dataset>/<target>_labeled_*.txt`) is used as labeled target data; `5` is the SSDAR setting of the paper.

Each command creates `results/<result-name>/experiment_0/`, trains for `--epochs` epochs, and
after every epoch evaluates on the target validation set. Whenever the validation loss improves,
the checkpoint is written and the model is also evaluated on the target test set — the reported
numbers are therefore the test metrics at the best validation epoch.

### Arguments

Defined in [utils/args.py](utils/args.py) and [main_proposed.py](main_proposed.py):

| Argument               | Type  | Default      | Description                                                                                                                                         |
| ---------------------- | ----- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--dataset`          | str   | `biwi`     | `biwi`, `qmul`, `mpi3d`                                                                                                                       |
| `--source`           | str   | `M`        | Source domain:`F`/`M` (BIWI, QMUL) or `real`/`realistic`/`toy` (MPI3D)                                                                    |
| `--target`           | str   | `F`        | Target domain, same choices as`--source`                                                                                                          |
| `--label-target-per` | str   | `5`        | Share of the target labeled pool used as labeled target data:`5`, `10`, `20`, `30`, `100` (→ `<target>_labeled_{5,10,20,30,full}.txt`) |
| `--model`            | str   | `resnet50` | Backbone: `resnet50`                                                                                                               |
| `--epochs`           | int   | `50`       | Number of training epochs                                                                                                                           |
| `--batch-size`       | int   | `48`       | Total batch size                              |
| `--lr`               | float | `1e-4`     | Learning rate of the AdamW optimizer                                                                                                                |
| `--scaler`           | bool  | `True`     | Min-max normalization of the regression targets                                                                                                     |
| `--warm-up`          | int   | `20`       | Epochs before the consistency losses are fully ramped up                                                                                            |
| `--lambda-1`         | float | `0.1`      | λ₁, weight of the prediction consistency loss (`L_crp`)                                                                                         |
| `--lambda-2`         | float | `0.1`      | λ₂, weight of the feature variance consistency loss (`L_crf`)                                                                                   |
| `--seed`             | int   | `0`        | Random seed (0, 1, 2)                                                                                                                                         |
| `--cuda`             | int   | `0`        | GPU index                                                                                                                                           |
| `--result-name`      | str   | `test`     | Name of the result folder under`results/`                                                                                                         |

### Outputs

Each run writes to `results/<result-name>/experiment_<id>/`, where `<id>` is auto-incremented so
repeated runs never overwrite each other:

```
results/<result-name>/experiment_<id>/
├── arg_parser.txt                     # all arguments + best epoch and valid/test MSE, R2, MAE
├── best_model.pth                     # checkpoint at the best validation epoch
├── csvs/preds_labels_<phase>_<epoch>.csv
├── plot/
└── events.out.tfevents.*              # TensorBoard scalars (mse, r2, mae per epoch)
```

Progress can be followed with `tensorboard --logdir results/`.

## License

See [LICENSE.txt](LICENSE.txt).
