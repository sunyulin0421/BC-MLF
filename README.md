# BC-MLF: Prediction-Layer Branch-Calibration for Multimodal Sentiment Analysis

This repository contains the training and evaluation code for BC-MLF. The method calibrates prediction-layer branches in a multimodal language model and is evaluated on MOSEI and SIMS.

## Setup

Create the environment:

```bash
conda env create -f environment.yml -n bcmlf
conda activate bcmlf
```

The data, language-model, and checkpoint roots are configurable. If unset, the repository-local directories are used:

```bash
export BCMLF_DATA_ROOT=/path/to/data/mmsa
export BCMLF_MODEL_ROOT=/path/to/models
export BCMLF_CHECKPOINT_ROOT=/path/to/checkpoints
```

The repository does not include datasets, language-model weights, or
checkpoints. Place the pre-extracted MMSA feature files under
`BCMLF_DATA_ROOT`, language models under `BCMLF_MODEL_ROOT` when using local
paths, and audiovisual checkpoints under `BCMLF_CHECKPOINT_ROOT`. The helper
scripts for preparing HF datasets and downloading language models are in
`scripts/`.

```bash
python scripts/prepare_mms2s.py --data-root "$BCMLF_DATA_ROOT"
python scripts/download_lms.py
```

## Training

Train the audiovisual encoder when a new encoder checkpoint is required:

```bash
python experiments/regression/mult_base.py \
  -m bienc -d mosei -g 0 \
  -c MMSA/config/regression/bcmlf/mosei/bienc.json \
  --exp-name bienc-mosei \
  -s 1990
```

Train BC-MLF with the supplied best configuration:

```bash
python experiments/regression/mult_base.py \
  -m msalm -d mosei -g 0 \
  -c MMSA/config/regression/bcmlf/mosei/large_best_bchead_eps008.json \
  --exp-name bcmlf-mosei \
  -s 1990 -s 1991
```

For SIMS, use `MMSA/config/regression/bcmlf/sims/base_best_bchead_eps008.json`.

## Evaluation

The training entry point performs normal test evaluation after training. To evaluate an existing checkpoint, pass `--eval_mode eval` and `--model_load_path`:

```bash
python experiments/regression/mult_base.py \
  -m msalm -d mosei \
  -c MMSA/config/regression/bcmlf/mosei/large_best_bchead_eps008.json \
  --eval_mode eval \
  --model_load_path "$BCMLF_CHECKPOINT_ROOT/your_model.pth" \
  -s 1990
```

## Repository layout

- `experiments/regression/`: training and evaluation entry points.
- `MMSA/models/`: model implementations, including `MSALM` and the audiovisual encoder.
- `MMSA/trains/`: optimization and evaluation loops.
- `MMSA/config/regression/bcmlf/`: BC-MLF training and evaluation configurations.
- `MMSA/data_loader.py`: multimodal feature and dataset loading.
- `environment.yml` and `requirements.txt`: runtime dependencies.

## Citation

```bibtex
@article{sun_bcmlf,
  title={Prediction-Layer Branch-Calibration for Multimodal Sentiment Analysis},
  author={Sun, Yulin and Xu, Kele and Dou, Yong},
}
```
