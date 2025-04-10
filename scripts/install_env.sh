#!/bin/bash
# shellcheck disable=SC2090,SC2086,SC2089,SC1091
# Default project path
PROJECT_PATH="$HOME/projects/photon"

# Parse command-line options
if ! OPTIONS=$(getopt -o p: --long project_path: -n 'parse-options' -- "$@"); then
	echo "install_env.sh: Error parsing options" >&2
	exit 1
fi

eval set -- "$OPTIONS"

echo "install_env.sh: options parsed OPTIONS=$OPTIONS"

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
printf "install_env.sh: arguments=%s, first argument=%s\n" "$@" "$1"
echo "install_env.sh: Install env in PROJECT_PATH=$PROJECT_PATH"
#! Moving to the project folder
cd "$PROJECT_PATH" || exit
#! Activate Poetry environment
POETRY_ENV_PATH=$(poetry env info --path)
if [[ -e $POETRY_ENV_PATH ]]; then
	echo "install_env.sh: Poetry environment exists."
	if ! [[ $(poetry check --lock) ]]; then
		echo "install_env.sh: Poetry environment is not up-to-date, updating..."
		poetry lock
	fi
	poetry run pre-commit install
else
	echo "install_env.sh: Poetry environment doesn't exist. Installing..."
	poetry config installer.max-workers 10
	poetry install -q
	POETRY_ENV_PATH=$(poetry env info --path)
	poetry run pre-commit install
fi
# shellcheck disable=SC1091
. "$POETRY_ENV_PATH"/bin/activate
# Adding CUDA paths to environment variables
export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

if [[ $1 == "no_cuda" ]]; then
	echo "install_env.sh: skipping any GPU-related setting."
else
	#! Check the output of `nvcc -V`
	NVCC_OUTPUT=$(nvcc -V)
	if [[ $NVCC_OUTPUT == *"release 12.4"* ]]; then
		echo "install_env.sh: CUDA 12.4 is detected."
	else
		echo "install_env.sh: CUDA 12.4 not detected. Please install CUDA 12.4. Exiting..."
		exit 1
	fi
	#! Install `flash-attn`
	if ! poetry run pip list | grep "flash_attn"; then
		echo "install_env.sh: Installing flash-attn..."
		poetry run pip install -q flash-attn==2.6.3 --no-build-isolation
		poetry run pre-commit install
	else
		echo "install_env.sh: flash-attn is already installed."
	fi
	#! Final message
	echo "install_env.sh: Environment is ready."
fi

# Set TRITON_CACHE_DIR to be system dependent, as such must be under '/home/<username>'
TRITON_CACHE_DIR="$PROJECT_PATH/.triton_cache"
export TRITON_CACHE_DIR

# Set the tmeporary directory for Ray
RAY_TEMP_DIR="/tmp/ray"
export RAY_TEMP_DIR
