# Query-Contrastive Grasp Fields (QCGF)

Language-driven grasp detection on Grasp-Anything++. Given an RGB image and an
English instruction, predict one grasp rectangle `(x, y, w, h, theta)`.
Paper: [paper/[FFI26] Hung_Tran.pdf](paper/%5BFFI26%5D%20Hung_Tran.pdf).

## Method

Frozen ImageNet ResNet-18 (strides 8/16/32) and frozen CLIP ViT-B/32 text
features feed a small trainable decoder (0.16M parameters): feature pyramid,
coordinates, FiLM text conditioning, and five 56x56 maps (quality,
cos 2θ, sin 2θ, width, height). The highest smoothed quality peak is decoded
into the grasp.

QCGF changes only the quality loss: foreground and background focal terms are
averaged separately, and a ranking term scores the target's grasp region above
grasp regions of other objects in the same training image. Distractor labels
are used in training only.

## Results

Scene-disjoint subset of Grasp-Anything++ (2,502 train / 280 val / 1,376 seen
test / 601 unseen test queries). Top-1 success (%), mean ± SD over seeds 7, 17, 27.

| Method | Seen | Unseen | H |
|---|---:|---:|---:|
| GR-ConvNet + CLIP (LGD author code) | 15.1 ± 5.0 | 12.0 ± 6.5 | 13.2 ± 6.0 |
| GR-ConvNet + CLIP, encoder trained | 14.3 ± 8.2 | 11.6 ± 11.6 | 12.4 ± 10.4 |
| MSE field | 39.0 ± 1.3 | 47.4 ± 3.1 | 42.7 ± 2.0 |
| Balanced BCE field | 48.2 ± 1.2 | 42.2 ± 8.2 | 44.7 ± 4.7 |
| Balanced focal field | 45.1 ± 1.8 | 46.4 ± 3.5 | 45.7 ± 2.5 |
| QCGF (balanced focal + ranking) | 50.6 ± 1.8 | 44.8 ± 4.1 | 47.4 ± 1.6 |

QCGF minus control in H (95% paired scene/seed bootstrap): MSE +4.6 [1.3, 8.2],
balanced BCE +2.7 [−2.7, 9.1], balanced focal +1.7 [−2.2, 5.8].

- QCGF beats the MSE field in every seed. Most of that gain comes from class
  balancing; the ranking term's extra effect is not significant with 3 seeds.
- GR-ConvNet + CLIP learns its visual features from 2.5k queries (the released
  code even detaches them), so its gap reflects pretraining, not the loss. Two
  trained-encoder seeds stopped early (patience 10); that row is a lower bound.
- The BCE/focal/GR-ConvNet controls were added after the first MSE/QCGF test
  results had been seen. Checkpoints are always chosen on validation.

## Setup

```bash
conda create -n qcgf python=3.10 -y
conda activate qcgf
pip install -r requirements.txt
```

For a GPU, install the CUDA build of `torch`/`torchvision` first (pytorch.org).

## Run

All commands run from the repository root. Script defaults are the reported
settings (grasp fields: 30 epochs, batch 32, AdamW 1e-3, rank weight 0.2;
GR-ConvNet: up to 50 epochs, patience 10, batch 8, Adam 1e-3).

```bash
# 1. Data subset and cached image/text features (CLIP downloads automatically)
python src/prepare_data.py --train-scenes 1000 --val-scenes 120
python src/cache_features.py

# 2. Grasp fields: 4 quality losses x 3 seeds, then test
for s in 7 17 27; do for m in mse bce focal qcgf; do
  python src/train.py --method $m --seed $s --out runs/${m}_seed$s
  python src/evaluate.py --checkpoint runs/${m}_seed$s/best.pt
done; done

# 3. GR-ConvNet + CLIP baseline (GPU recommended)
for s in 7 17 27; do
  python src/train_grconvnet.py --seed $s --out runs/grconvnet_clip_seed$s
  python src/train_grconvnet.py --train-encoder --seed $s \
    --out runs/grconvnet_clip_trained_encoder_seed$s
done

# 4. Results table and paired bootstrap
python src/summarize.py

# Predict on any image
python src/predict.py --image photo.jpg --text "Pick up the mug by its handle." \
  --checkpoint runs/qcgf_seed7/best.pt
```

## Layout

- `src/qcgf/model.py`: decoder and all quality losses (`mse`, `bce`, `focal`, `qcgf`).
- `src/qcgf/geometry.py`: target maps, decoding, success metric.
- `src/qcgf/data.py`: manifest and cached features; distractors come from the same image.
- `src/qcgf/grconvnet.py`: GR-ConvNet + CLIP from the
  [LGD repository](https://github.com/Fsoft-AIC/LGD) (MIT, commit `501904d`);
  changes are listed at the top of the file.
- `src/*.py`: data preparation, training, evaluation, summary, prediction.

## Data

Grasp-Anything++ is downloaded from the authors' Hugging Face repositories at
pinned revisions (`configs/dataset_sources.json`); raw data are not
redistributed.
