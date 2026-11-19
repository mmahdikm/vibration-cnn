# Vibration CNN

Small and large convolutional classifiers for six-channel temporal signals, with a synthetic-data training example.

Original development snapshot: **2022-02-08** (last recorded development commit).  
Clean public release: **2026-11-19**.

This repository starts with one public commit containing a cleaned snapshot of earlier work. The development date records the available source history; it is not a new claim of a formal project completion date.

## Architecture

`model.py` preserves the original temporal-convolution architecture: a wide input convolution, progressively narrower convolutional blocks, and a three-class classifier. The model accepts `(batch, 6, 1, 1300)` tensors. `CNN1D(size="small")` and `CNN1D(size="large")` select the two variants; `out_feature=True` also returns features.

The public release contains the model and a new portable training runner. It excludes private datasets, project notebooks, and optional analysis helpers whose redistribution permission could not be established.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --epochs 2
python train.py --size large --epochs 1
```

The default run generates six-channel sine waves for three synthetic classes. It demonstrates a working training pipeline, not real-world classification performance.

To train on your own data, create an NPZ file with `x` shaped `(N, 6, 1300)` and `y` shaped `(N,)` containing class IDs 0, 1, or 2:

```bash
python train.py --data signals.npz --epochs 20 --save artifacts/model.pt
```

The default is CPU. Use `--device cuda` on a suitable CUDA installation. There is no built-in held-out evaluation; split your own dataset and evaluate separately before drawing performance conclusions.

## License

Apache-2.0 for original code and public-release additions. The torchvision BSD notice is preserved for residual-block conventions; see `NOTICE` and `licenses/`.
