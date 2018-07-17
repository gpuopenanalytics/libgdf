#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Cristhian Alberto Gonzales Castillo <cristhian@blazingdb.com>
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

cmake_minimum_required(VERSION 2.8.12)

project(parquetcpp-download NONE)

include(ExternalProject)

set(PARQUET_VERSION "apache-parquet-cpp-1.4.0")

if (NOT $ENV{PARQUET_VERSION} STREQUAL "")
    set(PARQUET_VERSION $ENV{PARQUET_VETSION})
endif()

message(STATUS "Using Apache ParquetCpp version: ${PARQUET_VERSION}")

ExternalProject_Add(parquetcpp
    BINARY_DIR "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/parquetcpp-build"
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=RELEASE
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/parquetcpp-install
        -DPARQUET_ARROW_LINKAGE=static
        -DPARQUET_BUILD_SHARED=OFF
        -DPARQUET_BUILD_TESTS=OFF
    GIT_REPOSITORY git@github.com:apache/parquet-cpp.git
    GIT_TAG ${PARQUET_VERSION}
    INSTALL_DIR "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/parquetcpp-install"
    SOURCE_DIR "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/parquetcpp-src"
)
