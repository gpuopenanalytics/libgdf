ARG CUDA_VERSION=9.2
ARG UBUNTU_VERSION=16.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG CONDA_USER=conda
ARG GTEST_VERSION=1.8.1

# Set environment
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/lib

## Locked to Pandas 0.20.3 by https://github.com/gpuopenanalytics/pygdf/issues/118
ENV PANDAS_VERSION=0.20.3

#The basics - TODO: get rid of git in the image
RUN apt update -y --fix-missing && \
    apt upgrade -y && \
    apt install -y \
      git vim curl \
      libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/conda/bin:$PATH"
SHELL ["/bin/bash", "-c"]

ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda.sh 
ADD conda_environments/dev_py35.yml dev_py35.yml
RUN sh ./miniconda.sh -b -p /conda && \
    conda update -n base conda && \
    conda config --set report_errors false && \
    conda env create -n libgdf -f dev_py35.yml

# gtest depenencies stage
#ADD https://github.com/google/googletest/archive/release-${GTEST_VERSION}.tar.gz /tmp/googletest.tar.gz
#RUN mkdir -p googletest/build && \
#    tar -C googletest -xzf /tmp/googletest.tar.gz && \
#    cd googletest/build && \
#    /bin/bash -c "source /conda/etc/profile.d/conda.sh && \
#                  conda activate libgdf ; \
#                  cmake -DCMAKE_PREFIX_PATH=/conda/envs/libgdf/x86_64-conda_cos6-linux-gnu/sysroot \
#                        ../googletest-release-${GTEST_VERSION} ; \
#                  export DESTDIR=/conda/envs/libgdf/x86_64-conda_cos6-linux-gnu/sysroot && \
#                  make && make install" && \
#    cd .. && rm -rf googletest && rm /tmp/googletest.tar.gz


# Change to a non-root user so that it executes with minimum privilege
RUN groupadd --gid 3000 conda && \
    useradd -g conda -ms /bin/bash ${CONDA_USER} && \
    cat /conda/etc/profile.d/conda.sh >> /home/${CONDA_USER}/.bashrc && \
    chgrp -R conda /conda && \
    chmod -R g+rwx /conda
WORKDIR /home/${CONDA_USER}
USER ${CONDA_USER}


#RUN or ENTRYPOINT ?


