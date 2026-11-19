# Pixel-Level Anomaly Detection

**Spacepresso challenge · 3rd place · Leaderboard Pixel-AP: 0.9264**

An Advanced Deep Learning course project at Politecnico di Milano, developed by team **πthons**: **Daniele Iania, Alberto Ondei, Shuvro Ahmed and Umme Humaira Himi**.

The task was to localise defects across eight object categories by predicting a continuous anomaly score for every pixel. Our final approach combines **online synthetic defect generation**, **supervised segmentation trained separately for each object category**, and a **CNN–Transformer ensemble**.

[Project report](./Pixel-Level%20Anomaly%20Detection_Report.pdf) · [Colab notebook](./sourceCode/notebooks/colab_final_pithons.ipynb) · [Core implementation](./sourceCode/src/adl_anomaly/segmentation.py)

## Problem and motivation

The challenge provides five 224 × 224 views per sample, with pixel masks for the labelled anomalies. According to the report, each category contains approximately 2,000–2,600 normal images but only 4–7 labelled anomaly types, each represented by one defective sample. Defects vary from scratches and stains to missing parts and structural changes, and usually occupy a small fraction of the image.

We initially explored a convolutional autoencoder and PatchCore. Their score maps did not distinguish defective pixels from normal appearance reliably enough for this task. We then used the available masks to formulate the problem as **supervised binary segmentation**, generating additional defect–mask pairs during training to address the shortage of positive examples.

The dataset is multi-view, but the final notebook predicts each view independently. Optional multi-view post-processing exists in the source and is not enabled in the final blend.

## Method

### 1. Online defect generation

Training samples are constructed on demand from normal images and the available real anomalies:

| Training branch | Sampling probability |
| --- | ---: |
| Real anomaly with its annotated mask | 20% |
| Normal image with an empty mask | 15% |
| Synthetic scratches, blobs or structural copy-paste | 25% |
| Synthetic missing or chipped regions | 20% |
| Texture copy-paste from a real annotated anomaly | 20% |

These are sampling probabilities, not fixed proportions of a saved augmented dataset. Estimated object masks guide defect placement, and the generated modifications supply their own supervision masks. Training also uses horizontal flips, small affine transformations, brightness/contrast changes, colour jitter, Gaussian blur and ImageNet normalisation.

### 2. Per-category segmentation models

Three model families are trained separately for each object category:

| ID | Architecture | Backbone / initialisation | Training input | Loss |
| --- | --- | --- | --- | --- |
| A | U-Net | ImageNet-pretrained EfficientNet-B4 | 224 × 224 | 0.7 Lovász + 0.3 focal |
| B | DeepLabV3+ | ImageNet-pretrained EfficientNet-B5 | 224 × 224 | 0.7 Lovász + 0.3 focal |
| S | SegFormer | MiT-B2, initialised from an ADE20K-finetuned checkpoint | 512 × 512 | 0.5 BCE + 0.5 Dice |

Each model produces one anomaly channel. The focal loss uses α = 0.25 and γ = 2. The SegFormer classification head is replaced with a single-channel output.

Training uses AdamW, cosine learning-rate decay, mixed precision and checkpoint selection by validation loss. The notebook configures the CNNs for up to 20 epochs with learning rate 3e-4 and early-stopping patience 4; SegFormer uses up to 5 epochs with learning rate 2e-5 and patience 2. The validation limitations are described below.

### 3. Test-time augmentation and fusion

- **CNNs:** original, horizontal flip, vertical flip and combined flip, at scales 1.0, 1.33 and 1.66.
- **SegFormer:** the same four flip variants, at scales 1.0 and 1.33.
- Predictions are restored to a common orientation and resolution before averaging.
- Models A and B are combined using the code's `uncertainty` weighting: each model receives an image-level weight based on how far its probabilities are, on average, from 0.5. This is a confidence heuristic, not a learned uncertainty estimator.
- Each branch receives Gaussian smoothing, small-component filtering and HSV-based foreground masking to suppress background responses.
- The final submission blends **55% CNN ensemble + 45% SegFormer**.

The notebook uses Gaussian σ = 1.5 and removes components smaller than 16 pixels when identified above score threshold 0.1; the separate low-score cutoff is disabled. Maps remain continuous rather than becoming binary segmentation masks.

Each branch is exported in the challenge's **q8rle** format. The final blend decodes these maps, combines them with the stated weights and re-encodes the result.

## Results and evaluation context

**Competition placement: 3rd.** The report records a best official leaderboard **Pixel Average Precision of 0.9264**.

| Approach | Reported Pixel-AP | Evaluation setting |
| --- | ---: | --- |
| Convolutional autoencoder | ≈0.10 | Baseline validation estimate |
| PatchCore | ≈0.50 | Baseline validation estimate |
| Final supervised ensemble | **0.9264** | Official leaderboard submission |

The baseline estimates and leaderboard result come from different evaluation settings and should not be interpreted as a controlled improvement on one shared test split. Pixel-AP is average precision over pixel-level anomaly scores, not classification accuracy.

The report also compares the individual model families on **200 generated synthetic anomalies per category**. Their mean Pixel-AP values are 0.9599 (A), 0.9578 (B) and 0.9541 (S). These are synthetic-data diagnostics, separate from the official leaderboard result.

## Repository contents

| File | Purpose |
| --- | --- |
| [Project report](./Pixel-Level%20Anomaly%20Detection_Report.pdf) | Problem analysis, methodology, reported results and references |
| [Colab notebook](./sourceCode/notebooks/colab_final_pithons.ipynb) | Training, inference and final submission workflow |
| [segmentation.py](./sourceCode/src/adl_anomaly/segmentation.py) | Synthetic defects, models, losses, training, TTA and prediction |
| [data.py](./sourceCode/src/adl_anomaly/data.py) | Dataset discovery and image–mask pairing |
| [postprocess.py](./sourceCode/src/adl_anomaly/postprocess.py) | Smoothing, component filtering and foreground masking |
| [q8rle.py](./sourceCode/src/adl_anomaly/q8rle.py) | Continuous-map submission encoding and decoding |
| [cli.py](./sourceCode/src/adl_anomaly/cli.py) | Training, evaluation, prediction and submission-blending commands |
| [pyproject.toml](./sourceCode/pyproject.toml) / [requirements.txt](./sourceCode/requirements.txt) | Package configuration and dependencies |

The published code contains the final supervised pipeline. The autoencoder and PatchCore baselines are discussed in the report; their implementations are not included in this folder.

## Running the project

### Requirements and installation

Use Python 3.10 or newer and a compatible PyTorch environment. A CUDA GPU is recommended for training; the supplied Colab notebook explicitly requires a GPU runtime. The challenge dataset and trained checkpoints are not included in this repository.

From the repository root:

```bash
cd "Advanced Deep Learning/sourceCode"
python -m pip install -e .
spacepresso smoke-q8rle
```

The smoke test checks the submission encoder/decoder only; it does not train or evaluate the models.

### Dataset layout

Set `--data` to the directory containing `class_01` through `class_08`. Within each category, the loader expects:

| Relative path | Contents |
| --- | --- |
| `class_XX/train/good/` | Normal training images |
| `class_XX/train/anomaly_YY/` | Real defective training images |
| `class_XX/ground_truth_train/anomaly_YY/` | Matching annotation masks |
| `class_XX/test/` | Test images |

An anomaly image and its mask must have the same filename in the corresponding directories.

### Colab workflow

Open [colab_final_pithons.ipynb](./sourceCode/notebooks/colab_final_pithons.ipynb) in Colab and select a GPU runtime.

The setup cell assumes that the **contents of `sourceCode/`** are available at `/content/drive/MyDrive/ADL/`, with `pyproject.toml`, `requirements.txt` and `src/` directly inside that directory. It expects `dataset.zip` in the same location, containing a top-level `dataset/` folder. The notebook extracts the archive under `/content`, producing `/content/dataset`.

Update `project_dir`, `dataset_zip`, `DATA_DIR` and `CHECKPOINT_BACKUP_DIR` to match your environment. Then run setup, CNN training, SegFormer training, prediction and blending in order. If continuing from saved weights, configure the checkpoint paths instead of assuming the original team's Drive files are available.

### CLI training example

From `sourceCode/`, this trains both CNN families for one category using the final notebook's main training settings. Set `--data` to your dataset root. Omit `--classes class_01` to process all available categories.

```bash
spacepresso train-seg \
  --data /path/to/dataset \
  --classes class_01 \
  --seg-models A B \
  --work-dir outputs_seg_ab \
  --device cuda \
  --image-size 224 \
  --batch-size 32 \
  --num-workers 0 \
  --epochs 20 \
  --seg-loss hybrid \
  --early-stopping-patience 4
```

The five data-generation probabilities default to the values listed above. See the notebook for the SegFormer configuration, final TTA settings and 0.55/0.45 blend. Available commands are `train-seg`, `evaluate-seg`, `predict-seg`, `blend-submissions` and `smoke-q8rle`; each exposes `--help`.

## Limitations and reproducibility notes

- **Validation:** the implementation holds out 10% of normal images but reuses the same labelled anomaly pool in training and validation. Validation loss is therefore a model-selection diagnostic, not an independent estimate of performance on unseen real defects. `evaluate-seg` also evaluates annotated training anomalies.
- **Checkpoint selection:** models are selected by validation loss rather than a held-out Pixel-AP metric. A separate anomaly holdout would provide a stronger evaluation protocol.
- **Spatial detail:** the CNNs train at 224 × 224; very small or thin defects may benefit from higher-resolution training.
- **Foreground masking:** the HSV heuristic depends on the image background and lighting, and can suppress true defects near object boundaries. Small-component removal can also remove genuine tiny defects.
- **Reproduction:** dependency versions are not pinned, and neither the dataset nor the original model weights are included. The leaderboard score is a reported competition result, not a guarantee for a fresh training run.
- **Notebook outputs:** some stored training outputs come from earlier interrupted or erroneous commands and do not match the current cells. They should not be treated as a clean execution log of the complete workflow.

## Figures and references

The [report](./Pixel-Level%20Anomaly%20Detection_Report.pdf) contains the original project figures: **Figure 3, page 2** shows synthetic defects with their supervision masks; **Figure 4, page 3** shows a real anomaly, ground truth, individual predictions and the fused result. Figure 4 is a qualitative example, not an independent test benchmark.

The bibliography documents the methods that informed the project, including PatchCore, U-Net, DeepLabV3+, EfficientNet, SegFormer, Lovász/focal losses and synthetic-anomaly approaches such as DRAEM, CutPaste and NSA.
