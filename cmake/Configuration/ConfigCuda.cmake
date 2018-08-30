
set(CUDA_LIBRARY_DIR ${CUDA_TOOLKIT_ROOT_DIR})

## Based on NVRTC (Runtime Compilation) - CUDA Toolkit Documentation - v9.2.148
## 2.2. Installation
if(CMAKE_HOST_APPLE)
    set(CUDA_LIBRARY_DIR ${CUDA_TOOLKIT_ROOT_DIR}/lib)
elseif(CMAKE_HOST_UNIX)
    set(CUDA_LIBRARY_DIR ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
elseif(CMAKE_HOST_WIN32)
    set(CUDA_LIBRARY_DIR ${CUDA_TOOLKIT_ROOT_DIR}\lib\x64)
endif()

message("-- CUDA_LIBRARY_DIR: " ${CUDA_LIBRARY_DIR})
