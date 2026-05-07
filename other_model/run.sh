export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DISABLE_SENTRY=1
export CUDA_LAUNCH_BLOCKING=1

# 训练参数配置
LR=1e-4
MAX_HISTORY_LEN=50
EPOCH_PER_ROUND=10
TRAIN_BATCH_SIZE=256
EVAL_BATCH_SIZE=32
MIN_SEQ_LEN=20
MAX_SEQ_LEN=50
MODEL=TIGER
DATA_MODE=streaming
MODEL_SIZE=small
# 创建输出目录（若不存在）
mkdir -p runs/streaming

accelerate launch --mixed_precision=bf16 --num_processes=8 main_streaming.py \
  --model=${MODEL} \
  --json_config=./config/streaming_config.json \
  --model_size=${MODEL_SIZE} \
  --data_mode=${DATA_MODE} \
  --min_seq_len=${MIN_SEQ_LEN} \
  --max_seq_len=${MAX_SEQ_LEN} \
  --max_history_len=${MAX_HISTORY_LEN} \
  --lr=${LR} \
  --num_beams=50 \
  --epochs_per_round=${EPOCH_PER_ROUND} \
  --ndigits=8 \
  --codebook_size=512 \
  --save_every_round \
  --max_train_samples_per_user=999 \
  --use_wandb \
  --wandb_project=tiger-streaming \
  --wandb_run_name=exp_lr${LR}_epoch${EPOCH_PER_ROUND} \
  --train_batch_size=${TRAIN_BATCH_SIZE} \
  --eval_batch_size=${EVAL_BATCH_SIZE} \
  --output_dir=./outputs/${DATA_MODE}_train_test_${MODEL}_${MODEL_SIZE}_lr${LR}_epoch${EPOCH_PER_ROUND}_seqlen_max${MAX_SEQ_LEN}_min${MIN_SEQ_LEN}_bs${TRAIN_BATCH_SIZE} \
  > runs/${DATA_MODE}/train_test_${MODEL}_${MODEL_SIZE}_lr${LR}_epoch${EPOCH_PER_ROUND}_seqlen_max${MAX_SEQ_LEN}_min${MIN_SEQ_LEN}_bs${TRAIN_BATCH_SIZE}_$(date +%Y%m%d_%H%M).txt 2>&1

# accelerate launch --mixed_precision=bf16 --num_processes=8 main_streaming.py \
#   --model=TIGER \
#   --json_config=./config/streaming_config.json \
#   --model_size=tiny \
#   --min_seq_len=${MIN_SEQ_LEN} \
#   --max_seq_len=${MAX_SEQ_LEN} \
#   --max_history_len=${MAX_HISTORY_LEN} \
#   --lr=${LR} \
#   --num_beams=50 \
#   --epochs_per_round=${EPOCH_PER_ROUND} \
#   --ndigits=8 \
#   --codebook_size=512 \
#   --save_every_round \
#   --max_train_samples_per_user=999 \
#   --use_wandb \
#   --wandb_project=tiger-streaming \
#   --wandb_run_name=exp_lr${LR}_epoch${EPOCH_PER_ROUND} \
#   --train_batch_size=${TRAIN_BATCH_SIZE} \
#   --eval_batch_size=${EVAL_BATCH_SIZE} \
#   --output_dir=./outputs/streaming_train_test_lr${LR}_epoch${EPOCH_PER_ROUND}_seqlen_max${MAX_SEQ_LEN}_min${MIN_SEQ_LEN}_bs${TRAIN_BATCH_SIZE} \
#   > runs/streaming/train_test_lr${LR}_epoch${EPOCH_PER_ROUND}_seqlen_max${MAX_SEQ_LEN}_min${MIN_SEQ_LEN}_bs${TRAIN_BATCH_SIZE}_$(date +%Y%m%d_%H%M).txt 2>&1

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export DISABLE_SENTRY=1
# export CUDA_LAUNCH_BLOCKING=1

# # 训练参数配置
# LR=1e-4
# MAX_HISTORY_LEN=50
# EPOCH_PER_ROUND=1
# TRAIN_BATCH_SIZE=256
# EVAL_BATCH_SIZE=32
# MIN_SEQ_LEN=20
# MAX_SEQ_LEN=50

# # 创建输出目录（若不存在）
# mkdir -p runs/streaming

# accelerate launch --mixed_precision=bf16 --num_processes=8 main_streaming.py \
#   --model=TIGER \
#   --json_config=./config/streaming_config.json \
#   --model_size=tiny \
#   --min_seq_len=${MIN_SEQ_LEN} \
#   --max_seq_len=${MAX_SEQ_LEN} \
#   --max_history_len=${MAX_HISTORY_LEN} \
#   --lr=${LR} \
#   --num_beams=50 \
#   --epochs_per_round=${EPOCH_PER_ROUND} \
#   --ndigits=8 \
#   --codebook_size=512 \
#   --save_every_round \
#   --max_train_samples_per_user=999 \
#   --use_wandb \
#   --wandb_project=tiger-streaming \
#   --wandb_run_name=exp_lr${LR}_epoch${EPOCH_PER_ROUND} \
#   --train_batch_size=${TRAIN_BATCH_SIZE} \
#   --eval_batch_size=${EVAL_BATCH_SIZE} \
#   --output_dir=./outputs/streaming_train_test_lr${LR}_epoch${EPOCH_PER_ROUND}_seqlen_max${MAX_SEQ_LEN}_min${MIN_SEQ_LEN}_bs${TRAIN_BATCH_SIZE} \
#   > runs/streaming/train_test_lr${LR}_epoch${EPOCH_PER_ROUND}_seqlen_max${MAX_SEQ_LEN}_min${MIN_SEQ_LEN}_bs${TRAIN_BATCH_SIZE}_$(date +%Y%m%d_%H%M).txt 2>&1

# accelerate launch --mixed_precision=bf16 --num_processes=8 main_streaming.py \
#   --model=TIGER \
#   --json_config=./config/streaming_config.json \
#   --model_size=tiny \
#   --min_seq_len=${MIN_SEQ_LEN} \
#   --max_seq_len=${MAX_SEQ_LEN} \
#   --max_history_len=${MAX_HISTORY_LEN} \
#   --lr=${LR} \
#   --num_beams=50 \
#   --epochs_per_round=${EPOCH_PER_ROUND} \
#   --ndigits=8 \
#   --codebook_size=512 \
#   --save_every_round \
#   --max_train_samples_per_user=999 \
#   --use_wandb \
#   --wandb_project=tiger-streaming \
#   --wandb_run_name=exp_lr${LR}_epoch${EPOCH_PER_ROUND} \
#   --train_batch_size=${TRAIN_BATCH_SIZE} \
#   --eval_batch_size=${EVAL_BATCH_SIZE} \
#   --output_dir=./outputs/streaming_train_test_lr${LR}_epoch${EPOCH_PER_ROUND}_seqlen_max${MAX_SEQ_LEN}_min${MIN_SEQ_LEN}_bs${TRAIN_BATCH_SIZE} \
#   > runs/streaming/train_test_lr${LR}_epoch${EPOCH_PER_ROUND}_seqlen_max${MAX_SEQ_LEN}_min${MIN_SEQ_LEN}_bs${TRAIN_BATCH_SIZE}_$(date +%Y%m%d_%H%M).txt 2>&1

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export DISABLE_SENTRY=1
# export CUDA_LAUNCH_BLOCKING=1

# # 训练参数配置
# LR=1e-4
# MAX_HISTORY_LEN=50
# EPOCH_PER_ROUND=1
# TRAIN_BATCH_SIZE=256
# EVAL_BATCH_SIZE=32
# MIN_SEQ_LEN=20
# MAX_SEQ_LEN=50

# # 创建输出目录（若不存在）
# mkdir -p runs/streaming

# accelerate launch --mixed_precision=bf16 --num_processes=8 main_streaming.py \
#   --model=TIGER \
#   --json_config=./config/streaming_config.json \
#   --model_size=tiny \
#   --min_seq_len=${MIN_SEQ_LEN} \
#   --max_seq_len=${MAX_SEQ_LEN} \
#   --max_history_len=${MAX_HISTORY_LEN} \
#   --lr=${LR} \
#   --num_beams=50 \
#   --epochs_per_round=${EPOCH_PER_ROUND} \
#   --ndigits=8 \
#   --codebook_size=512 \
#   --save_every_round \
#   --max_train_samples_per_user=999 \
#   --use_wandb \
#   --wandb_project=tiger-streaming \
#   --wandb_run_name=exp_lr${LR}_epoch${EPOCH_PER_ROUND} \
#   --train_batch_size=${TRAIN_BATCH_SIZE} \
#   --eval_batch_size=${EVAL_BATCH_SIZE} \
#   --output_dir=./outputs/streaming_train_test_lr${LR}_epoch${EPOCH_PER_ROUND}_seqlen_max${MAX_SEQ_LEN}_min${MIN_SEQ_LEN}_bs${TRAIN_BATCH_SIZE} \
#   > runs/streaming/train_test_lr${LR}_epoch${EPOCH_PER_ROUND}_seqlen_max${MAX_SEQ_LEN}_min${MIN_SEQ_LEN}_bs${TRAIN_BATCH_SIZE}_$(date +%Y%m%d_%H%M).txt 2>&1

# accelerate launch --mixed_precision=bf16 --num_processes=8 main_streaming.py \
#   --model=TIGER \
#   --json_config=./config/streaming_config.json \
#   --model_size=tiny \
#   --min_seq_len=${MIN_SEQ_LEN} \
#   --max_seq_len=${MAX_SEQ_LEN} \
#   --max_history_len=${MAX_HISTORY_LEN} \
#   --lr=${LR} \
#   --num_beams=50 \
#   --epochs_per_round=${EPOCH_PER_ROUND} \
#   --ndigits=8 \
#   --codebook_size=512 \
#   --save_every_round \
#   --max_train_samples_per_user=999 \
#   --use_wandb \
#   --wandb_project=tiger-streaming \
#   --wandb_run_name=exp_lr${LR}_epoch${EPOCH_PER_ROUND} \
#   --train_batch_size=${TRAIN_BATCH_SIZE} \
#   --eval_batch_size=${EVAL_BATCH_SIZE} \
#   --output_dir=./outputs/streaming_train_test_lr${LR}_epoch${EPOCH_PER_ROUND}_seqlen_max${MAX_SEQ_LEN}_min${MIN_SEQ_LEN}_bs${TRAIN_BATCH_SIZE} \
#   > runs/streaming/train_test_lr${LR}_epoch${EPOCH_PER_ROUND}_seqlen_max${MAX_SEQ_LEN}_min${MIN_SEQ_LEN}_bs${TRAIN_BATCH_SIZE}_$(date +%Y%m%d_%H%M).txt 2>&1

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export DISABLE_SENTRY=1
# export CUDA_LAUNCH_BLOCKING=1

# # 训练参数配置
# LR=1e-4
# MAX_HISTORY_LEN=50
# EPOCH_PER_ROUND=1
# TRAIN_BATCH_SIZE=256
# EVAL_BATCH_SIZE=32
# MIN_SEQ_LEN=20
# MAX_SEQ_LEN=50

# # 创建输出目录（若不存在）
# mkdir -p runs/streaming

# accelerate launch --mixed_precision=bf16 --num_processes=8 main_streaming.py \
#   --model=TIGER \
#   --json_config=./config/streaming_config.json \
#   --model_size=tiny \
#   --min_seq_len=${MIN_SEQ_LEN} \
#   --max_seq_len=${MAX_SEQ_LEN} \
#   --max_history_len=${MAX_HISTORY_LEN} \
#   --lr=${LR} \
#   --num_beams=50 \
#   --epochs_per_round=${EPOCH_PER_ROUND} \
#   --ndigits=8 \
#   --codebook_size=512 \
#   --save_every_round \
#   --max_train_samples_per_user=999 \
#   --use_wandb \
#   --wandb_project=tiger-streaming \
#   --wandb_run_name=exp_lr${LR}_epoch${EPOCH_PER_ROUND} \
#   --train_batch_size=${TRAIN_BATCH_SIZE} \
#   --eval_batch_size=${EVAL_BATCH_SIZE} \
#   --output_dir=./outputs/streaming_train_test_lr${LR}_epoch${EPOCH_PER_ROUND}_seqlen_max${MAX_SEQ_LEN}_min${MIN_SEQ_LEN}_bs${TRAIN_BATCH_SIZE} \
#   > runs/streaming/train_test_lr${LR}_epoch${EPOCH_PER_ROUND}_seqlen_max${MAX_SEQ_LEN}_min${MIN_SEQ_LEN}_bs${TRAIN_BATCH_SIZE}_$(date +%Y%m%d_%H%M).txt 2>&1

# accelerate launch --mixed_precision=bf16 --num_processes=8 main_streaming.py \
#   --model=TIGER \
#   --json_config=./config/streaming_config.json \
#   --model_size=tiny \
#   --min_seq_len=${MIN_SEQ_LEN} \
#   --max_seq_len=${MAX_SEQ_LEN} \
#   --max_history_len=${MAX_HISTORY_LEN} \
#   --lr=${LR} \
#   --num_beams=50 \
#   --epochs_per_round=${EPOCH_PER_ROUND} \
#   --ndigits=8 \
#   --codebook_size=512 \
#   --save_every_round \
#   --max_train_samples_per_user=999 \
#   --use_wandb \
#   --wandb_project=tiger-streaming \
#   --wandb_run_name=exp_lr${LR}_epoch${EPOCH_PER_ROUND} \
#   --train_batch_size=${TRAIN_BATCH_SIZE} \
#   --eval_batch_size=${EVAL_BATCH_SIZE} \
#   --output_dir=./outputs/streaming_train_test_lr${LR}_epoch${EPOCH_PER_ROUND}_seqlen_max${MAX_SEQ_LEN}_min${MIN_SEQ_LEN}_bs${TRAIN_BATCH_SIZE} \
#   > runs/streaming/train_test_lr${LR}_epoch${EPOCH_PER_ROUND}_seqlen_max${MAX_SEQ_LEN}_min${MIN_SEQ_LEN}_bs${TRAIN_BATCH_SIZE}_$(date +%Y%m%d_%H%M).txt 2>&1


