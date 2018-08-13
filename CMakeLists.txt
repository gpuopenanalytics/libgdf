#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
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

# Usage:
#    $ mkdir build       # create directory for out-of-source build
#    $ cd build          # enter the build directory
#    $ cmake ..          # configure build system
#    $ make              # make libgdf
#    $ make pytest       # trigger test
#    $ make pytest-ci    # trigger test with jUnit.xml output
#    $ make install      # install libgdf

PROJECT(libgdf)

cmake_minimum_required(VERSION 2.8)  # not sure about version required

set(CMAKE_CXX_STANDARD 11)
message(STATUS "Using C++ standard: c++${CMAKE_CXX_STANDARD}")

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})
message(STATUS "CMAKE_MODULE_PATH:" "${CMAKE_MODULE_PATH}")

# Include CMake modules
include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)
include(CTest)

# Include custom modules (see cmake directory)
include(ConfigureGoogleTest)
include(ConfigureArrow)

# Find CUDA and detect hardware-specific flags
find_package(CUDA)
set_package_properties(
    CUDA PROPERTIES TYPE REQUIRED
    PURPOSE "NVIDIA CUDA® parallel computing platform and programming model."
    URL "https://developer.nvidia.com/cuda-zone")

if (CUDA_FOUND)
  # Get CUDA architecture flag "--generate-code=arch=compute_XX,code=sm_XX"
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CUDA-Utils/bin)

  set(OUTPUTFILE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CUDA-Utils/bin/get_cuda_arch)
  set(CUDAFILE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CUDA-Utils/get_cuda_arch.cu)
  execute_process(COMMAND nvcc -lcuda ${CUDAFILE} -o ${OUTPUTFILE})
  execute_process(COMMAND ${OUTPUTFILE}
                  RESULT_VARIABLE CUDA_RETURN_CODE
                  OUTPUT_VARIABLE DETECTED_ARCH)

  if(${CUDA_RETURN_CODE} EQUAL 0)
    set(CUDA_SUCCESS "TRUE")
  else()
    set(CUDA_SUCCESS "FALSE")
  endif()

  if (${CUDA_SUCCESS})
    message(STATUS "Detected CUDA Architecture: ${DETECTED_ARCH}")
    message(STATUS "Detected CUDA Version: ${CUDA_VERSION_STRING}")
    message(STATUS "Detected CUDA Path: ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "Detected CUDA Libararies: ${CUDA_LIBRARIES}")
    message(STATUS "Detected CUDA Performance Primitives: ${CUDA_npp_LIBRARY}")

  else()
    message(WARNING ${DETECTED_ARCH})
  endif()

  # Get CUDA compute capability "X.X"
  set(OUTPUTFILE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CUDA-Utils/bin/get_cuda_cc)
  set(CUDAFILE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CUDA-Utils/get_cuda_cc.cu)
  execute_process(COMMAND nvcc -lcuda ${CUDAFILE} -o ${OUTPUTFILE})
  execute_process(COMMAND ${OUTPUTFILE}
                  RESULT_VARIABLE CUDA_RETURN_CODE
                  OUTPUT_VARIABLE CUDA_COMPUTE_CAPABILITY)

  if(${CUDA_RETURN_CODE} EQUAL 0)
    set(CUDA_SUCCESS "TRUE")
  else()
    set(CUDA_SUCCESS "FALSE")
  endif()

  if (${CUDA_SUCCESS})
    message(STATUS "Detected CUDA Compute Capability: ${CUDA_COMPUTE_CAPABILITY}")
  endif()
endif()

# Option handling for specifying target GPU Architecture
set(GPU_ARCH detected CACHE STRING "Specify the target GPU Architecture")
string(TOLOWER "${GPU_ARCH}" GPU_ARCH)

set(PASCAL_ARCH --generate-code=arch=compute_61,code=sm_61)
set(VOLTA_ARCH  --generate-code=arch=compute_70,code=sm_70)

if(${GPU_ARCH} STREQUAL detected)
  message(STATUS "Using detected GPU Architecture")
  message(STATUS "Setting CUDA NVCC FLAGS: ${CUDA_NVCC_FLAGS}")
  
  if(${CUDA_COMPUTE_CAPABILITY} LESS 6.0) # Full functionality available in Pascal or greater
    message(WARNING "\n Detected CUDA comptue capability: ${CUDA_COMPUTE_CAPABILITY}
                     \n NOT ALL FEATURES WILL FUNCTION PROPERLY")
  endif()
  
  set(CUDA_NVCC_FLAGS ${DETECTED_ARCH};--expt-extended-lambda;-std=c++11)
endif()

if(${GPU_ARCH} STREQUAL pascal)
  message(STATUS "Setting GPU Arch: Pascal")
  set(CUDA_NVCC_FLAGS ${PASCAL_ARCH};--expt-extended-lambda;-std=c++11)
  message(STATUS "Setting CUDA NVCC FLAGS: ${CUDA_NVCC_FLAGS}")
endif()

if(${GPU_ARCH} STREQUAL volta)
  message(STATUS "Setting GPU Arch: Volta")
  set(CUDA_NVCC_FLAGS ${VOLTA_ARCH};--expt-extended-lambda;-std=c++11)
  message(STATUS "Setting CUDA NVCC FLAGS: ${CUDA_NVCC_FLAGS}")
endif()

# Option handling for CUDA debugging
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-g;-G)
endif()

# Option handling for joins
option(HASH_JOIN "Hash-based implementation for joins" ON)
if(HASH_JOIN)
  message(STATUS "Hash-based implementation for joins: ${HASH_JOIN}")
endif()

# Option handling for allocator to be used with hash tables
option(HT_LEGACY_ALLOCATOR "Enable legacy allocator for hash tables" ON)
if(HT_LEGACY_ALLOCATOR)
  message(STATUS "Legacy allocator for hash tables: ${HT_LEGACY_ALLOCATOR}")
endif()

# Locate the Apache Arrow package (Requires that you use ConfigureArrow module)
message(STATUS "ARROW_ROOT: " ${ARROW_ROOT})
find_package(Arrow REQUIRED)
set_package_properties(Arrow PROPERTIES TYPE REQUIRED
    PURPOSE "Apache Arrow is a cross-language development platform for in-memory data."
    URL "https://github.com/apache/arrow")

if (ARROW_FOUND)
    message(STATUS "Apache Arrow found in ${ARROW_ROOT}")
else()
    message(FATAL_ERROR "Apache Arrow not found, please check your settings.")
endif()

include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub"
    "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/moderngpu/src"
    "${CUDA_INCLUDE_DIRS}"
    "${ARROW_INCLUDEDIR}"
)

cuda_add_library(gdf SHARED
    src/binaryops.cu
    src/bitmaskops.cu
    src/column.cpp
    src/cudautils.cu
    src/datetimeops.cu
    src/errorhandling.cpp
    src/filterops.cu
    src/hashing.cu
    src/hashops.cu
    src/ipc.cu
    src/joining.cu
    src/reductions.cu
    src/scan.cu
    src/segmented_sorting.cu
    src/sorting.cu
    src/sqls_ops.cu
    src/streamcompactionops.cu
    src/unaryops.cu
    #src/windowedops.cu
)

if (ARROW_NUMERIC_VERSION EQUAL 900)
  # arrow-0.9 requires boost_regex
  target_link_libraries(gdf arrow boost_regex)
else()
  target_link_libraries(gdf arrow)
endif()

# Command to symlink files into the build directory
add_custom_command(  # link the include directory
    OUTPUT include
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/include include
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_custom_command(  # link the python directory
    OUTPUT libgdf_cffi setup.py tests
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/python/libgdf_cffi libgdf_cffi
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/python/tests tests
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.py setup.py)

add_custom_command(  # trigger cffi to build the wrapper
    OUTPUT libgdf_cffi/libgdf_cffi.py
    COMMAND python setup.py build_ext --inplace
    DEPENDS setup.py libgdf_cffi include)

add_custom_target(   # target to link the python files and trigger cffi
    copy_python
    DEPENDS libgdf_cffi/libgdf_cffi.py)

# The test target
add_custom_target(pytest DEPENDS copy_python)
add_custom_command(TARGET pytest POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CMAKE_BINARY_DIR} py.test -v
                   WORKING_DIRECTORY --cache-clear tests)

# The test CI target with junit.xml output
add_custom_target(pytest-ci DEPENDS copy_python)
add_custom_command(TARGET pytest POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CMAKE_BINARY_DIR} py.test -v
                   WORKING_DIRECTORY --cache-clear --junitxml=junit.xml tests)

# The install target
install(TARGETS gdf LIBRARY DESTINATION lib)
install(DIRECTORY include/gdf DESTINATION include)

# Configure the C++ tests
find_package(GTest QUIET)
set_package_properties(GTest PROPERTIES TYPE OPTIONAL
    PURPOSE "Google C++ Testing Framework (Google Test)."
    URL "https://github.com/google/googletest")

if(GTEST_FOUND)
    message(STATUS "Google C++ Testing Framework (Google Test) found in ${GTEST_ROOT}")
    include_directories(${GTEST_INCLUDE_DIRS})
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/tests)
else()
    message(AUTHOR_WARNING "Google C++ Testing Framework (Google Test) not found: automated tests are disabled.")
endif()

# Print the project summary
feature_summary(WHAT ALL INCLUDE_QUIET_PACKAGES FATAL_ON_MISSING_REQUIRED_PACKAGES)
