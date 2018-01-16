#!/bin/bash
#
# Adapted from install-cuda-trusty.sh
#
# Install the core CUDA toolkit for a ubuntu-xenial (16.04) system. Requires the
# CUDA environment variable to be set to the required version.
#
# Since this script updates environment variables, to execute correctly you must
# 'source' this script, rather than executing it in a sub-process.
#
set -e
apt-get update -qq
apt-get install -y wget git
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_${CUDA}_amd64.deb
dpkg -i cuda-repo-ubuntu1604_${CUDA}_amd64.deb
apt-get update -qq
export CUDA_APT=${CUDA:0:3}
export CUDA_APT=${CUDA_APT/./-}
# sudo apt-get install -y cuda-drivers cuda-core-${CUDA_APT} cuda-cudart-dev-${CUDA_APT} cuda-cufft-dev-${CUDA_APT}
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-drivers cuda-core-${CUDA_APT} cuda-cudart-dev-${CUDA_APT}
apt-get clean
export CUDA_HOME=/usr/local/cuda-${CUDA:0:3}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export PATH=${CUDA_HOME}/bin:${PATH}
