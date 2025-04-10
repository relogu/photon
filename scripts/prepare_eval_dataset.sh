#!/bin/bash
# shellcheck disable=SC2090,SC2086,SC2089,SC1091
## This script aims to prepare the evaluation gauntlet datasets. The datasets will be downloaded
## from the Lorenzo's fork of the `llm-foundry` repository.

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

export DATASET_CACHE="$PROJECT_PATH"

#! Check what has been set in the environmental variables
echo "prepare_eval_dataset.sh: DATASET_CACHE=$DATASET_CACHE"
echo "prepare_eval_dataset.sh: the datasets will be moved to $DATASET_CACHE/eval/local_data"

#! Clone the repo and move to the 'fl' branch
cd $PROJECT_PATH || exit
cd ..
git clone git@github.com:relogu/llm-foundry.git
cd llm-foundry || exit
git fetch origin
git checkout --track origin/fl

#! Copy the datasets folder to the DATASET_CACHE folder
mkdir -p "$DATASET_CACHE"/eval/local_data
cp -r scripts/eval/local_data/* "$DATASET_CACHE"/eval/local_data
