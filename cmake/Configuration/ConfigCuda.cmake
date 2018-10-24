#=============================================================================
# Copyright 2018-2019 BlazingDB, Inc.
#     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

set(CUDA_LIBRARY_DIR       ${CUDA_TOOLKIT_ROOT_DIR})
set(CUDA_LIBRARY_STUBS_DIR ${CUDA_TOOLKIT_ROOT_DIR})

## Based on NVRTC (Runtime Compilation) - CUDA Toolkit Documentation - v9.2.148
## 2.2. Installation
if(CMAKE_HOST_APPLE)
    set(CUDA_LIBRARY_DIR       ${CUDA_TOOLKIT_ROOT_DIR}/lib)
    set(CUDA_LIBRARY_STUBS_DIR ${CUDA_TOOLKIT_ROOT_DIR}/lib/stubs)
elseif(CMAKE_HOST_UNIX)
    set(CUDA_LIBRARY_DIR       ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    set(CUDA_LIBRARY_STUBS_DIR ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs)
elseif(CMAKE_HOST_WIN32)
    set(CUDA_LIBRARY_DIR       ${CUDA_TOOLKIT_ROOT_DIR}\lib\x64)
    set(CUDA_LIBRARY_STUBS_DIR ${CUDA_TOOLKIT_ROOT_DIR}\lib\x64\stubs)
endif()

set(CUDA_CUDA_LIB  cuda)
set(CUDA_NVRTC_LIB nvrtc)

message(STATUS "CUDA_LIBRARY_DIR: ${CUDA_LIBRARY_DIR}")
message(STATUS "CUDA_LIBRARY_STUBS_DIR: ${CUDA_LIBRARY_STUBS_DIR}")
