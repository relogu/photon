#!/bin/bash
# shellcheck disable=SC2090,SC2086,SC2089,SC1091

# Default project path
PROJECT_PATH="$HOME/projects/photon"
# Get the current date and time
DATETIME=$(date '+%Y%m%d_%H%M%S')

# Parse command-line options
if ! OPTIONS=$(getopt -o p: --long project_path: -n 'parse-options' -- "$@"); then
	echo "fed_125m_example.sh: Error parsing options" >&2
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
echo "fed_125m_example.sh: PROJECT_PATH=$PROJECT_PATH"

cd $PROJECT_PATH || exit

LOCAL_BATCH_SIZE=32
LOCAL_STEPS=128
N_CLIENTS=8
TOTAL_STEPS=$((5120 * 256 / (LOCAL_BATCH_SIZE)))
WARMUP_STEPS=$((100 * 256 / (LOCAL_BATCH_SIZE)))
N_ROUNDS=$((TOTAL_STEPS / (LOCAL_STEPS)))
FL_FL_EVAL_PERIOD=null
export RUN_UUID="fed-125m-example-$DATETIME"

EXTERNAL_CONFIGS=""
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.max_duration=${TOTAL_STEPS}ba"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.scheduler.schedulers.lr.t_max=${TOTAL_STEPS}ba"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.scheduler.schedulers.lr.t_warmup=${WARMUP_STEPS}ba"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.scheduler.schedulers.lr.alpha_f=0.1"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.optimizer.lr=6.0e-4"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS fl.strategy_kwargs.server_learning_rate=1.0"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS fl.strategy_kwargs.server_momentum=0.0"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS fl.n_rounds=$N_ROUNDS"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.local_steps=${LOCAL_STEPS}ba"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS ++llm_config.device_eval_microbatch_size=auto"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS fl.reset_optimizer=false"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.save_interval=${LOCAL_STEPS}ba"

#! Dataset configs
DATASET_NAME="fed-c4"
export DATASET_CACHE_DIR="$PROJECT_PATH"
mkdir -p $DATASET_CACHE_DIR
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS dataset=$DATASET_NAME"                                           # Dataset name
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS dataset.train.root_local=$DATASET_CACHE_DIR/$DATASET_NAME"       # Path of the local cache for the training dataset
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS dataset.val.root_local=$DATASET_CACHE_DIR/$DATASET_NAME"         # Path of the local cache for the evaluation dataset
#! If the number of clients is 1, use the single-client dataset
if [ $N_CLIENTS -eq 1 ]; then
	EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS dataset/streams@dataset.train.streams=1_client_small" # Stream configuration for the training dataset
	EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS dataset/streams@dataset.val.streams=1_client_small"   # Stream configuration for the training dataset
else
	EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS dataset/streams@dataset.train.streams=${N_CLIENTS}_clients" # Stream configuration for the training dataset
	EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS dataset/streams@dataset.val.streams=${N_CLIENTS}_clients"   # Stream configuration for the training dataset
fi
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS centralized.stream_id=null"

#! Batch size configs
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.global_train_batch_size=$LOCAL_BATCH_SIZE"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.device_train_microbatch_size=auto"
# EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.device_train_microbatch_size=16"
# EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.device_train_microbatch_size=8"
# EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.device_train_microbatch_size=4"
# EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.device_train_microbatch_size=1"

#! Precision and attention implementation configs
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.precision=amp_bf16"
# EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.precision=amp_fp16"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.model.attn_config.attn_impl=flash"
# EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.model.attn_config.attn_impl=torch"
# EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS ~llm_config.fsdp_config" # DDP

EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS fl.n_total_clients=$N_CLIENTS"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS fl.n_clients_per_round=$N_CLIENTS"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS fl.eval_period=$FL_FL_EVAL_PERIOD"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.eval_interval=${TOTAL_STEPS}ba"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS photon.checkpoint=false"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS photon.comm_stack.shm=false"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS photon.comm_stack.ray=true"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS use_wandb=false"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS cleanup_checkpoints=false"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS cleanup_checkpoints_per_round=false"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.eval_first=false"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS ~llm_config.loggers.wandb"
EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS ~llm_config.loggers.tensorboard"

# Resume options
# EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS  photon.resume_round=-1"

# External configs will be appended at the end of the command, thus rule over those
# in the `photon_llm_125M.sh` script
export EXTERNAL_CONFIGS

# Set the Flower driver and fleet API addresses
FLOWER_SUPERLINK_IP=$(hostname -I | awk '{print $1}')
#! Set the Flower driver and fleet API addresses if they haven't been set
DRIVER_API_ADDRESS=${DRIVER_API_ADDRESS:-"$FLOWER_SUPERLINK_IP:54752"}
FLEET_API_ADDRESS=${FLEET_API_ADDRESS:-"$FLOWER_SUPERLINK_IP:54753"}

# Set Ray address
RAY_PORT=51550
RAY_NODE_IP=$(hostname -I | awk '{print $1}')
RAY_ADDRESS="$RAY_NODE_IP:$RAY_PORT"

#! Exporting the variables for the `photon_llm_125M.sh` script
export DRIVER_API_ADDRESS
export FLEET_API_ADDRESS
export RAY_ADDRESS
export RAY_PORT
export RAY_NODE_IP

# The following command will start the Ray head node. When running in a single node
# setup, it is sufficient as both client and server will automatically attach to the
# current Ray session.
poetry run ray start --head --port=$RAY_PORT &

bash $PROJECT_PATH/scripts/photon_llm_125M.sh -p "$PROJECT_PATH" 125M

# NOTE: For executing in a multinode setup, the Ray head node will need to be started
# only on the first node (e.g., `ray start --head --port=$RAY_PORT`), which can run the
# federated learning server, the Flower Superlink, and, potentially, a single client
#(e.g., `bash $PROJECT_PATH/scripts/photon_llm_125M.sh -p "$PROJECT_PATH" 125M`). The other nodes
# that run, let's say, other clients (e.g., `DRIVER_API_ADDRESS="$FLOWER_SUPERLINK_IP:54752" FLEET_API_ADDRESS="$FLOWER_SUPERLINK_IP:54753" photon_llm_125M_client_only.sh`,
# where `$FLOWER_SUPERLINK_IP` is the IP address of the first node), will need to attach
# to the Ray head node (e.g., `ray start --address=$RAY_ADDRESS`, where `$RAY_ADDRESS`
# is the IP address and port of the head node) and run the `photon_llm_125M_client_only.sh`
# script.
