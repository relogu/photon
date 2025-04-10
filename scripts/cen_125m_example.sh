#!/bin/bash
# shellcheck disable=SC2090,SC2086,SC2089,SC1091

# Default project path
PROJECT_PATH="$HOME/projects/photon"
# Get the current date and time
DATETIME=$(date '+%Y%m%d_%H%M%S')

# Parse command-line options
if ! OPTIONS=$(getopt -o p: --long project_path: -n 'parse-options' -- "$@"); then
	echo "cen_125m_example.sh: Error parsing options" >&2
	exit 1
fi

eval set -- "$OPTIONS"

while true; do
	case "$1" in
	-p | --project_path)
		PROJECT_PATH="$2"
		shift 2
		;;
	--)
		shift
		break
		;;
	*)
		break
		;;
	esac
done
echo "cen_125m_example.sh: PROJECT_PATH=$PROJECT_PATH"

#! Fix the number of total samples/tokens to train on
BATCH_SIZE=256
N_STEPS=5120
echo "Training with batch size: $BATCH_SIZE"
WARMUP_STEPS=100
COOLDOWN_STEPS=240
EVAL_INTERVAL=$((N_STEPS / 100))
export RUN_UUID="cen-125M-bs$BATCH_SIZE-$DATETIME"

#! Set the number of clients
N_CLIENTS=8

#! Initialize the external configs
EXTERNAL_CONFIGS=""
#! Set configs related to learning rate
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.max_duration=${N_STEPS}ba"
# EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS ++llm_config.callbacks.noise_scale_monitor={}"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.scheduler.schedulers.lr.name=constant_with_sqrt_cooldown_with_warmup"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.scheduler.schedulers.lr.t_max=${N_STEPS}ba"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.scheduler.schedulers.lr.t_warmup=${WARMUP_STEPS}ba"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS ++llm_config.scheduler.schedulers.lr.t_cooldown=${COOLDOWN_STEPS}ba"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS ~llm_config.scheduler.schedulers.lr.alpha_f"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS ~llm_config.algorithms.gradient_clipping"

#! DecoupledAdamW
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.optimizer.name=decoupled_adamw"

#! Dataset configs
DATASET_NAME="fed-c4"
export DATASET_CACHE_DIR="$PROJECT_PATH"
mkdir -p $DATASET_CACHE_DIR
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS dataset=$DATASET_NAME"                                     # Dataset name
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS dataset.train.root_local=$DATASET_CACHE_DIR/$DATASET_NAME" # Path of the local cache for the training dataset
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS dataset.val.root_local=$DATASET_CACHE_DIR/$DATASET_NAME"   # Path of the local cache for the evaluation dataset
#! If the number of clients is 1, use the single-client dataset
if [ $N_CLIENTS -eq 1 ]; then
	EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS dataset/streams@dataset.train.streams=1_client_small" # Stream configuration for the training dataset
	EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS dataset/streams@dataset.val.streams=1_client_small"   # Stream configuration for the training dataset
else
	EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS dataset/streams@dataset.train.streams=${N_CLIENTS}_clients" # Stream configuration for the training dataset
	EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS dataset/streams@dataset.val.streams=${N_CLIENTS}_clients"   # Stream configuration for the training dataset
fi
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS centralized.stream_id=null" # Setting null concatenates all streams for centralized training

#! Batch size configs
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.global_train_batch_size=$BATCH_SIZE"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.device_train_microbatch_size=auto"

#! Precision and attention implementation configs
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.precision=amp_bf16"
# EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.precision=amp_fp16"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.model.attn_config.attn_impl=flash"
# EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.model.attn_config.attn_impl=torch"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS ~llm_config.fsdp_config" # DDP

#! Additional configs
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.device_eval_batch_size=256"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.eval_subset_num_batches=1"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS ++llm_config.device_eval_microbatch_size=auto"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS ++llm_config.tp_config=null"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.eval_interval=${N_STEPS}ba"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.save_interval=${EVAL_INTERVAL}ba"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.eval_interval=${EVAL_INTERVAL}ba"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.eval_first=false"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.console_log_interval=100ba"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS ~llm_config.loggers.wandb"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS ~llm_config.loggers.tensorboard"

# External configs will be appended at the end of the command, thus rule over those in the `centralised_training.sh` script
export EXTERNAL_CONFIGS

bash $PROJECT_PATH/scripts/centralised_training.sh -p "$PROJECT_PATH" 125M
