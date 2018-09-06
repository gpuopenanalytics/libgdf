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

cmake_minimum_required(VERSION 3.0)

project(google-benchmark)

include(ExternalProject)

ExternalProject_Add(
    google-benchmark
    GIT_REPOSITORY    https://github.com/google/benchmark.git
    GIT_TAG           master
    SOURCE_DIR        "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/google-benchmark/source"
    BINARY_DIR        "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/google-benchmark/build"
    INSTALL_DIR       "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/google-benchmark/install"
    CMAKE_ARGS        -DCMAKE_BUILD_TYPE=Release
                      -DBENCHMARK_ENABLE_TESTING=OFF
                      -DBENCHMARK_ENABLE_GTEST_TESTS=OFF
                      -DBENCHMARK_DOWNLOAD_DEPENDENCIES=OFF
                      -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/google-benchmark/install
)