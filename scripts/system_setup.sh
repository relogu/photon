#!/bin/bash
# shellcheck disable=SC2090,SC2086,SC2089,SC1091
## This script aims to setup the OS for a (cloud) VM
## starting from the "Plain Ubuntu 22.04" image
#! Update and upgrade package manager
sudo apt-get update
#! Installing the essentials
sudo apt-get install -y build-essential zlib1g-dev libedit-dev \
	libssl-dev liblzma-dev libffi-dev libbz2-dev \
	libreadline-dev libsqlite3-dev bc
#! Check the output of `nvcc -V`
NVCC_OUTPUT=$(nvcc -V)
if [[ $NVCC_OUTPUT == *"release 12.4"* ]]; then
	echo "CUDA 12.4 is detected."
else
	#! Get and install CUDA 12.4.1 and its drivers
	wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
	sudo sh cuda_12.4.1_550.54.15_linux.run --toolkit --no-man-page --silent
fi
if [[ $PATH == *"cuda-12.4"* ]]; then
	echo "PATH variable is already set."
else
	#! Set the PATH env variables for the new CUDA version
	echo '# Adding CUDA 12.4 to the PATH environmental variables' >>~/.bashrc
	# shellcheck disable=SC2016
	echo 'export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}' >>~/.bashrc
	export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}
fi
if [[ $LD_LIBRARY_PATH == *"cuda-12.4"* ]]; then
	echo "LD_LIBRARY_PATH variable is already set."
else
	#! Set the LD_LIBRARY_PATH env variables for the new CUDA version
	# shellcheck disable=SC2016
	echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >>~/.bashrc
	export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
fi
#! Set GPU persistence mode
sudo nvidia-smi -pm 1

#! Install CuDNN latest
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cudnn9-cuda-12

#! Install `pyenv`
PYENV_VER_OUTPUT=$(pyenv --version)
if [[ $PYENV_VER_OUTPUT == *"pyenv "* ]]; then
	echo "pyenv is already installed."
else
	#! Getting `pyenv`
	curl https://pyenv.run | bash
fi
if [[ $PYENV_ROOT == *"pyenv"* ]]; then
	echo "PYENV_ROOT variable is already set."
else
	#! Setting up `pyenv` to execute automatically in the shell
	echo "# Load 'pyenv' automatically" >>~/.bashrc
	# shellcheck disable=SC2016
	echo 'export PYENV_ROOT="$HOME/.pyenv"' >>~/.bashrc
	export PYENV_ROOT="$HOME/.pyenv"
	# shellcheck disable=SC2016
	echo '[[ -d "$PYENV_ROOT"/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >>~/.bashrc
	[[ -d "$PYENV_ROOT"/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
	# shellcheck disable=SC2016
	echo 'eval "$(pyenv init -)"' >>~/.bashrc
	eval "$(pyenv init -)"
	echo '# # Load pyenv-virtualenv automatically' >>~/.bashrc
	# shellcheck disable=SC2016
	echo '# eval "$(pyenv virtualenv-init -)"' >>~/.bashrc
fi

#! Installing python 3.11.9
pyenv install -s 3.11.9
#! Selecting this python as global
pyenv global 3.11.9
#! Upgrade pip
pip install --upgrade pip
#! Monitoring utilities
sudo apt install bpytop
pip install nvitop
#! Install poetry
pip install poetry
#! Install cmake
pip install cmake

#! Set up S3 credentials (for using S3-backed streaming datasets)
mkdir ~/.aws
echo '[default]' >~/.aws/config
echo '[default]' >~/.aws/credentials
echo '    aws_access_key_id = <aws_access_key_id>' >>~/.aws/credentials
echo '    aws_secret_access_key = <aws_secret_access_key>' >>~/.aws/credentials

#! Set up wandb credentials (for using Wandb)
echo 'machine api.wandb.ai' >~/.netrc
echo '    login user' >>~/.netrc
echo '    password <password>' >>~/.netrc
