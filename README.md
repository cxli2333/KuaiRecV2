# KuaiRecV2

Official code for the KuaiRecV2 paper.

[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-KuaiRecV2-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/cxcx2333/KuaiRecV2)


## Structure

- `MiniOneRec/`: generative recommendation with SFT, RL, and EWC-LoRA.
- `MiniOneRec/streaming/`: streaming training, data preparation, and evaluation.
- `other_model/`: TIGER baseline and recommendation utilities.

## Installation

Use Python 3.10+ and a CUDA-compatible PyTorch build for GPU training.

```bash
pip install -r requirements.txt
```

For DeepSpeed RL training:

```bash
pip install deepspeed bitsandbytes
```

## Data preparation

Prepare daily interaction TSVs, item metadata with semantic IDs, and model checkpoints. Update the data paths in:

- `MiniOneRec/streaming/Full_Month_file_paths_relative.json`
- `other_model/config/streaming_config.json`

Place MiniOneRec's `master.item.json`, `master.index.json`, and `master.info.txt` in `MiniOneRec/streaming_data/info/`.

## Usage

Run commands from the indicated directories and configure model paths and GPU counts for your setup.

**TIGER streaming training**

```bash
cd other_model
bash run.sh
```

**EWC-LoRA streaming training**

```bash
cd MiniOneRec/streaming
bash streaming_ewc_lora.sh
```

**Behavior evaluation**

```bash
cd MiniOneRec/streaming
bash behavior_eval.sh
```

Additional entry points:

| Task | Script |
| --- | --- |
| SFT / RL | `MiniOneRec/sft.py`, `MiniOneRec/rl.py` |
| Streaming SFT / RL | `MiniOneRec/streaming/streaming_sft.py`, `streaming_rl.py`, `streaming_sft_rl.py` |
| Prediction evaluation | `MiniOneRec/streaming/streaming_eval.py` |
| Recommendation metrics | `MiniOneRec/metric.py`, `MiniOneRec/list_metric.py` |
| Checkpoint backtesting | `MiniOneRec/streaming/backtest_first_day_checkpoints.py` |

Sample logging is disabled by default. Keep credentials in environment variables and leaderboard ground-truth/contact repositories private.
