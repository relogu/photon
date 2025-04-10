#!/bin/bash
# shellcheck disable=SC2090,SC2086,SC2089,SC1091
# Default project path
PROJECT_PATH="$HOME/projects/photon"

# Parse command-line options
if ! OPTIONS=$(getopt -o p: --long project_path: -n 'parse-options' -- "$@"); then
	echo "set_llm_config.sh: Error parsing options" >&2
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
echo "set_llm_config.sh: PROJECT_PATH=$PROJECT_PATH"
#! Check if there's an input argument
if [[ $# -eq 0 ]]; then
	echo "set_llm_config.sh: No input argument supplied."
	exit 1
fi

#! Defaults - MPT models
LLM_CONFIG_MPT_125M="llm_config=mpt-125m"
LLM_CONFIG_MPT_1B="llm_config=mpt-1b"
LLM_CONFIG_MPT_3B="llm_config=mpt-3b"
LLM_CONFIG_MPT_7B="llm_config=mpt-7b"

#! Set the run configuration
if [[ $1 == "125M" ]]; then
	export LLM_CONFIG="$FLOP_COUNT $LLM_CONFIG_MPT_125M"
elif [[ $1 == "1B" ]]; then
	export LLM_CONFIG="$FLOP_COUNT $LLM_CONFIG_MPT_1B"
elif [[ $1 == "3B" ]]; then
	export LLM_CONFIG="$FLOP_COUNT $LLM_CONFIG_MPT_3B"
elif [[ $1 == "7B" ]]; then
	export LLM_CONFIG="$FLOP_COUNT $LLM_CONFIG_MPT_7B"
else
	echo "set_llm_config.sh: Invalid input argument: $1"
	echo "set_llm_config.sh: Valid input arguments are: 125M, 1B, 3B, 7B"
	exit 1
fi

echo "set_llm_config.sh: Selected LLM config: $1 ($LLM_CONFIG)"
printf "set_llm_config.sh: arguments=%s, first argument=%s\n" "$@" "$1"

#! Remove the positional arguments
eval set --
