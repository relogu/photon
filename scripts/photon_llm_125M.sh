#!/bin/bash
# shellcheck disable=SC2090,SC2086,SC2089,SC1091
# Default project path
PROJECT_PATH="$HOME/projects/photon"

# Parse command-line options
if ! OPTIONS=$(getopt -o p: --long project_path: -n 'parse-options' -- "$@"); then
	echo "Error parsing options" >&2
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

echo "PROJECT_PATH=$PROJECT_PATH"
#! Moving to the project folder
cd "$PROJECT_PATH" || exit
#! Preparing environment
#! Executing the environment preparation script
#! NOTE: Must use "." to execute, "sh" doesn't work
. "$PROJECT_PATH"/scripts/install_env.sh -p "$PROJECT_PATH"
#! Set `LLM_CONFIG` environment variable
. "$PROJECT_PATH"/scripts/set_llm_config.sh -p "$PROJECT_PATH" "125M"
#! Saving path
DATETIME=$(date '+%Y%m%d_%H%M%S')
#! If RUN_UUID hasn't been set, set it to the default value
if [ -z "$RUN_UUID" ]; then
	export RUN_UUID="fed-125M-$DATETIME"
fi
#! If SAVE_PATH hasn't been set, set it to the default value
if [ -z "$SAVE_PATH" ]; then
	export SAVE_PATH="$PROJECT_PATH/$RUN_UUID"
	mkdir -p "$SAVE_PATH"
fi
export PHOTON_SAVE_PATH="$PROJECT_PATH/runs/$RUN_UUID/$DATETIME"
mkdir -p "$PHOTON_SAVE_PATH"
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
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
#! Set Photon and FL config
N_LOCAL_STEPS=${N_LOCAL_STEPS:-1}
PHOTON_CONFIG="run_uuid=$RUN_UUID photon.refresh_period=100"

#! Photon configuration
PHOTON_CONFIG="$PHOTON_CONFIG run_uuid=$RUN_UUID"                # Run UUID
PHOTON_CONFIG="$PHOTON_CONFIG photon.checkpoint=false"           # Disable checkpointing
PHOTON_CONFIG="$PHOTON_CONFIG photon.checkpoint=true"            # Enable checkpointing
PHOTON_CONFIG="$PHOTON_CONFIG photon.saving_path=$SAVE_PATH"     # Save path for Photon Server
PHOTON_CONFIG="$PHOTON_CONFIG llm_config.save_folder=$SAVE_PATH" # Save path for PhotonLLM Client
PHOTON_CONFIG="$PHOTON_CONFIG llm_config.save_overwrite=true"    # Overwrite the existing checkpoint for PhotonLLM Client
PHOTON_CONFIG="$PHOTON_CONFIG photon.n_nodes=1"                  # Number of nodes in the Photon federation
PHOTON_CONFIG="$PHOTON_CONFIG photon.resume_round=-1"            # Resume round from the latest for the Photon Server
PHOTON_CONFIG="$PHOTON_CONFIG photon.resume_round=null"          # Start from scratch
PHOTON_CONFIG="$PHOTON_CONFIG photon.refresh_period=60"          # PhotonLLM Client workers refresh period
PHOTON_CONFIG="$PHOTON_CONFIG photon.restore_run_uuid=null"      # Restore run UUID for Photon Server

#! FL setting
PHOTON_CONFIG="$PHOTON_CONFIG fl.n_total_clients=1"     # Number of total clients in the federation
PHOTON_CONFIG="$PHOTON_CONFIG fl.n_clients_per_round=1" # Number of clients per round
PHOTON_CONFIG="$PHOTON_CONFIG fl.n_rounds=10"           # Total number of rounds

#! ServerOpt
PHOTON_CONFIG="$PHOTON_CONFIG fl.strategy_name=NESTOROV"                                                          # Server optimizer strategy
PHOTON_CONFIG="$PHOTON_CONFIG fl.strategy_kwargs.server_learning_rate=1.0 fl.strategy_kwargs.server_momentum=0.0" # FedAvg

#! ClientOpt (AdamW + Cosine LR scheduler) parameters
PHOTON_CONFIG="$PHOTON_CONFIG llm_config.scheduler.schedulers.lr.t_max=5000ba"
PHOTON_CONFIG="$PHOTON_CONFIG llm_config.scheduler.schedulers.lr.t_warmup=100ba"
PHOTON_CONFIG="$PHOTON_CONFIG llm_config.scheduler.schedulers.lr.alpha_f=0.1"
PHOTON_CONFIG="$PHOTON_CONFIG llm_config.optimizer.lr=6.0e-4"
PHOTON_CONFIG="$PHOTON_CONFIG fl.reset_optimizer=false" # Keep the local optimizer every round
PHOTON_CONFIG="$PHOTON_CONFIG fl.reset_optimizer=true"  # Reset local optimizer every round

#! Training hyperparameters
PHOTON_CONFIG="$PHOTON_CONFIG llm_config.save_interval=${N_LOCAL_STEPS}ba" # Save checkpoint interval
PHOTON_CONFIG="$PHOTON_CONFIG llm_config.save_interval=90000ba"            # Save checkpoint interval
PHOTON_CONFIG="$PHOTON_CONFIG llm_config.console_log_interval=100ba"       # Console log interval
PHOTON_CONFIG="$PHOTON_CONFIG llm_config.local_steps=${N_LOCAL_STEPS}ba"   # Local steps
PHOTON_CONFIG="$PHOTON_CONFIG llm_config.eval_first=true"                  # Enable evaluation at the first step
PHOTON_CONFIG="$PHOTON_CONFIG llm_config.eval_interval=250ba"              # Local evaluation interval
PHOTON_CONFIG="$PHOTON_CONFIG llm_config.eval_subset_num_batches=1"        # Evaluate only one batch of the entire validation set
PHOTON_CONFIG="$PHOTON_CONFIG llm_config.eval_subset_num_batches=1"        # Evaluate the entire validation set
# PHOTON_CONFIG="$PHOTON_CONFIG ~llm_config.fsdp_config"                     # Use DDP only
# PHOTON_CONFIG="$PHOTON_CONFIG ++llm_config.compile_config={}"                           # Compile the model at Trainer initialization
# PHOTON_CONFIG="$PHOTON_CONFIG ++llm_config.fsdp_config.sharding_strategy=SHARD_GRAD_OP" # Shard only the gradient operation -- most of the times convenient when GPUs are poorly connected
PHOTON_CONFIG="$PHOTON_CONFIG llm_config.precision=amp_fp16"                 # Fastest precision context when using DDP
PHOTON_CONFIG="$PHOTON_CONFIG ++llm_config.device_eval_microbatch_size=auto" # Automatic microbatch size for evaluation
PHOTON_CONFIG="$PHOTON_CONFIG llm_config.device_eval_batch_size=128"         # Evaluation batch size
PHOTON_CONFIG="$PHOTON_CONFIG llm_config.device_train_microbatch_size=auto"  # Automatic microbatch size

#! Model parameters
PHOTON_CONFIG="$PHOTON_CONFIG llm_config.model.attn_config.attn_impl=torch" # Use PyTorch's attention implementation
PHOTON_CONFIG="$PHOTON_CONFIG llm_config.model.attn_config.attn_impl=flash" # Use Flash attention implementation

#! Evaluation gauntlet configuration
# PHOTON_CONFIG="$PHOTON_CONFIG icl_tasks_config=tasks_v0.3 eval_gauntlet_config=eval_gauntlet_v0.3 eval_gauntlet_config.destination_dir=$DATASET_CACHE_DIR/eval icl_tasks_config.root_dir=$DATASET_CACHE_DIR" # Complete MosaicML Gauntlet
PHOTON_CONFIG="$PHOTON_CONFIG icl_tasks_config=empty eval_gauntlet_config=empty" # Empty gauntlet

#! Run Hydra resolver
HYDRA_FULL_ERROR=1 poetry run python -m photon.hydra_resolver $LLM_CONFIG $PHOTON_CONFIG $EXTERNAL_CONFIGS hydra/job_logging=none hydra/hydra_logging=none 2>&1 | tee "$PHOTON_SAVE_PATH"/hydra_resolver.log

#! Set the Flower driver and fleet API addresses if they haven't been set
DRIVER_API_ADDRESS=${DRIVER_API_ADDRESS:-"[::]:54752"}
FLEET_API_ADDRESS=${FLEET_API_ADDRESS:-"[::]:54753"}

#! Start a Superlink
# GRPC_VERBOSITY=debug
poetry run flower-superlink --insecure --driver-api-address "${DRIVER_API_ADDRESS}" --fleet-api-address "${FLEET_API_ADDRESS}" 2>&1 | tee "$PHOTON_SAVE_PATH"/superlink.log &
SUPERLINK_PID=$!
sleep 5

#! Launch ServerWithPhoton as a ServerApp
# GRPC_VERBOSITY=debug
poetry run flower-server-app photon.server_app:app --insecure --superlink "${DRIVER_API_ADDRESS}" 2>&1 | tee "$PHOTON_SAVE_PATH"/server.log &
#! Keep the pid of the ServerApp
SERVERAPP_PID=$!

sleep 5

#! Launch NodeManager as a SuperNode - ClientApp
#! NOTE: Adding `NCCL_BLOCKING_WAIT=1` breaks the optimizer's checkpointing. We don't know why yet.
# NCCL_DEBUG=INFO NCCL_NVB_DISABLE=1 NCCL_NVLS_ENABLE=0 # For running on Lambda Labs faulty machine
# GRPC_VERBOSITY=debug
CUDA_LAUNCH_BLOCKING=1 poetry run flower-client-app photon.client_app:app --insecure --superlink "${FLEET_API_ADDRESS}" --persist-client 2>&1 | tee "$PHOTON_SAVE_PATH"/node_manager.log &
#! Keep the pid of the ClientApp
CLIENTAPP_PID=$!

# Enable CTRL+C to stop all background processes
trap 'kill $CLIENTAPP_PID $SUPERLINK_PID $SERVERAPP_PID' SIGINT SIGTERM

#! Wait for the ServerApp to finish
wait $SERVERAPP_PID
#! Kill the ClientApp and Superlink
kill $CLIENTAPP_PID
kill $SUPERLINK_PID
