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

NUMBER_OF_TOTAL_CLIENTS=8

#! Execute the command
poetry run python -m photon.dataset.convert_dataset_hf \
	--path c4 \
	--name en \
	--splits train_small val_xxsmall \
	--tokenizer "EleutherAI/gpt-neox-20b" \
	--bos_text "<|endoftext|>" \
	--eos_text "<|endoftext|>" \
	--remote_path "fed-c4/c$NUMBER_OF_TOTAL_CLIENTS" \
	--num_clients $NUMBER_OF_TOTAL_CLIENTS \
	--concat_tokens 2048

#! Remove the positional arguments
eval set --
