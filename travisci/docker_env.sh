# CUDA variable needs to be set above
export CUDA_HOME=/usr/local/cuda-${CUDA:0:3}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export PATH=${CUDA_HOME}/bin:$HOME/miniconda3/bin:${PATH}
