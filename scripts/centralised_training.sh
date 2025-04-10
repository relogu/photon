#!/bin/bash
# shellcheck disable=SC2090,SC2086,SC2089,SC1091
# Default project path
PROJECT_PATH="$HOME/projects/photon"

# Parse command-line options
if ! OPTIONS=$(getopt -o p: --long project_path: -n 'parse-options' -- "$@"); then
	echo "centralised_training.sh: Error parsing options" >&2
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
MODEL_SIZE="$1"
echo "centralised_training.sh: MODEL_SIZE=$MODEL_SIZE"

#! Check if at least one arguments are passed
if [[ $# -lt 1 ]]; then
	echo "centralised_training.sh: Illegal number of parameters."
	echo "Usage: centralised_training.sh <llm_model_config>"
	exit 1
fi
echo "centralised_training.sh: PROJECT_PATH=$PROJECT_PATH"
#! Moving to the project folder
cd "$PROJECT_PATH" || exit
#! Executing the environment preparation script
#! NOTE: Must use "." to execute, "sh" doesn't work
. "$PROJECT_PATH"/scripts/install_env.sh -p "$PROJECT_PATH"
#! Set `LLM_CONFIG` environment variable
. "$PROJECT_PATH"/scripts/set_llm_config.sh -p "$PROJECT_PATH" "$MODEL_SIZE"
#! Saving path
DATETIME=$(date '+%Y%m%d_%H%M%S')
#! If RUN_UUID hasn't been set, set it to the default value
if [ -z "$RUN_UUID" ]; then
	export RUN_UUID="centralised-$MODEL_SIZE-$DATETIME"
fi
#! If SAVE_PATH hasn't been set, set it to the default value
if [ -z "$SAVE_PATH" ]; then
	export SAVE_PATH="$PROJECT_PATH/$RUN_UUID"
	mkdir -p "$SAVE_PATH"
fi
export PHOTON_SAVE_PATH="$PROJECT_PATH/runs/$RUN_UUID/$DATETIME"
mkdir -p "$PHOTON_SAVE_PATH"

#! Size specific optimization parameters (comment this out to use the default values)
LLM_OPTIONS="$LLM_OPTIONS llm_config.max_duration=10ba"                    # Number of total training steps
LLM_OPTIONS="$LLM_OPTIONS llm_config.scheduler.schedulers.lr.t_max=10ba"   # Duration of the learning rate cosine scheduler
LLM_OPTIONS="$LLM_OPTIONS llm_config.scheduler.schedulers.lr.t_warmup=1ba" # Warmup steps for the learning rate scheduler
LLM_OPTIONS="$LLM_OPTIONS llm_config.scheduler.schedulers.lr.alpha_f=0.1"  # Final learning rate multiplier for the cosine scheduler
LLM_OPTIONS="$LLM_OPTIONS llm_config.optimizer.lr=6.0e-4"                  # Learning rate

#! General training parameters
LLM_OPTIONS="$LLM_OPTIONS llm_config.save_interval=1000ba"       # Save interval
LLM_OPTIONS="$LLM_OPTIONS llm_config.console_log_interval=100ba" # Console log interval
LLM_OPTIONS="$LLM_OPTIONS llm_config.eval_first=false"           # Disable evaluation first
LLM_OPTIONS="$LLM_OPTIONS llm_config.eval_first=true"            # Enable evaluation first
LLM_OPTIONS="$LLM_OPTIONS llm_config.eval_interval=250ba"        # Local evaluation interval
LLM_OPTIONS="$LLM_OPTIONS llm_config.eval_subset_num_batches=1"  # Evaluate the entire validation set
# LLM_OPTIONS="$LLM_OPTIONS ++llm_config.compile_config={}"                             # Compiles the model with default parameters
# LLM_OPTIONS="$LLM_OPTIONS llm_config.fsdp_config.sharding_strategy=SHARD_GRAD_OP" # Shard only the gradient operation -- most of the times convenient when GPUs are poorly connected
# LLM_OPTIONS="$LLM_OPTIONS ~llm_config.fsdp_config" # Removes FSDP
# LLM_OPTIONS="$LLM_OPTIONS ~llm_config.callbacks.optimizer_monitor"                    # Clears OptimizerMonitor
# LLM_OPTIONS="$LLM_OPTIONS ~llm_config.callbacks.lr_monitor"                           # Clears LRMonitor
# LLM_OPTIONS="$LLM_OPTIONS ~llm_config.callbacks.memory_monitor"                       # Clears MemoryMonitor
# LLM_OPTIONS="$LLM_OPTIONS ~llm_config.callbacks.runtime_estimator"                    # Clears RuntimeEstimator
# LLM_OPTIONS="$LLM_OPTIONS ~llm_config.callbacks.activation_monitor_full_model"        # Clears ActivationMonitorFullModel
LLM_OPTIONS="$LLM_OPTIONS ++llm_config.device_eval_microbatch_size=auto" # Automatic microbatch size for evaluation
LLM_OPTIONS="$LLM_OPTIONS llm_config.device_eval_batch_size=128"         # Evaluation batch size
LLM_OPTIONS="$LLM_OPTIONS llm_config.global_train_batch_size=512"        # DisTrO 8 devices batch size
LLM_OPTIONS="$LLM_OPTIONS llm_config.global_train_batch_size=128"        # DisTrO 2 devices batch size
LLM_OPTIONS="$LLM_OPTIONS llm_config.global_train_batch_size=256"        # DisTrO 4 devices batch size
LLM_OPTIONS="$LLM_OPTIONS llm_config.global_train_batch_size=64"         # DisTrO single device batch size
LLM_OPTIONS="$LLM_OPTIONS llm_config.device_train_microbatch_size=8"     # DisTrO 4xA40 devices microbatch size -- w/ and w/o compilation -- no FSDP
LLM_OPTIONS="$LLM_OPTIONS llm_config.device_train_microbatch_size=auto"  # Automatic microbatch size
LLM_OPTIONS="$LLM_OPTIONS llm_config.precision=amp_fp16"                 # Standard Automatic Mixed Precision float16 context -- use with < Ampere GPUs
# LLM_OPTIONS="$LLM_OPTIONS llm_config.precision=amp_fp8 ++llm_config.model.fc_type=te" # Standard Automatic Mixed Precision float8 context -- use with >= Ampere GPUs
LLM_OPTIONS="$LLM_OPTIONS llm_config.precision=amp_bf16" # Standard Automatic Mixed Precision brainfloat16 precision context -- use with >= Ampere GPUs

#! Model parameters
# LLM_OPTIONS="$LLM_OPTIONS llm_config.model.attn_config.attn_impl=torch"                                                                                                                                                   # Use PyTorch's attention implementation
# LLM_OPTIONS="$LLM_OPTIONS llm_config.model.attn_config.attn_impl=flash"                                                                                                                                                   # Use flash attention implementation

#! Getting visible GPUs
if [[ $(nvidia-smi -L) == *'No devices'* ]]; then
	echo "No NVIDIA devices found."
	N_GPUS=0
elif [[ $(nvidia-smi -L) == *'not found'* ]]; then
	echo "nvidia-smi not present."
	N_GPUS=0
else
	N_GPUS=$(nvidia-smi -L | wc -l)
fi
if [ "$N_GPUS" -eq 0 ]; then
	echo "No GPUs found. Exiting."
	CUDA_VISIBLE_DEVICES=""
else
	CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS - 1)))
fi
echo "centralised_training.sh: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
#! Additional config
LLM_OPTIONS="$LLM_OPTIONS run_uuid=$RUN_UUID"

#! Evaluation gauntlet configuration
# LLM_OPTIONS="$LLM_OPTIONS icl_tasks_config=tasks_v0.3 eval_gauntlet_config=eval_gauntlet_v0.3 eval_gauntlet_config.destination_dir=$DATASET_CACHE_DIR/eval icl_tasks_config.root_dir=$DATASET_CACHE_DIR" # Complete MosaicML Gauntlet
LLM_OPTIONS="$LLM_OPTIONS icl_tasks_config=empty eval_gauntlet_config=empty" # Empty gauntlet

export LLM_OPTIONS

echo "centralised_training.sh: LLM_OPTIONS=$LLM_OPTIONS"

#! Run Hydra resolver
HYDRA_FULL_ERROR=1 poetry run python -m photon.hydra_resolver $LLM_CONFIG $LLM_OPTIONS $DATA_CONFIG $EXTERNAL_CONFIGS hydra/job_logging=none hydra/hydra_logging=none 2>&1 | tee "$PHOTON_SAVE_PATH"/hydra_resolver.log

#! Launch centralised training script
#! NOTE: Adding `NCCL_BLOCKING_WAIT=1` breaks the optimizer's checkpointing. We don't know why yet.
# TORCH_LOGS="+dynamo" TORCHDYNAMO_VERBOSE=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True APPOINTED_CUDA_DEVICE=$CUDA_VISIBLE_DEVICES CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 RUN_UUID=$(uuidgen) poetry run composer --world_size $N_GPUS --node_rank 0 --master_addr 127.0.0.1 $PROJECT_PATH/photon/centralised_train.py hydra/job_logging=none hydra/hydra_logging=none 2>&1 | tee $PHOTON_SAVE_PATH/centralised_train.log &
#! Keep the pid and wait for it
BACK_PID=$!
wait $BACK_PID
