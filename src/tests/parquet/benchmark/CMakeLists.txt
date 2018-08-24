#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Alexander Ocsa <alexander@blazingdb.com>
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

find_package(Boost REQUIRED COMPONENTS filesystem)

set(PARQUET_FILE_PATH
    ${CMAKE_SOURCE_DIR}/src/tests/parquet/benchmark/reader-test.parquet)

GDF_ADD_PARQUET_BENCHMARK(parquet-benchmark parquet-benchmark.cu
                          ../../helper/utils.cuh 
                          ../../helper/utils.cu
)

if (GDF_BENCHMARK)
target_compile_definitions(parquet-benchmark
    PUBLIC -DPARQUET_FILE_PATH="${PARQUET_FILE_PATH}")
endif()

# GDF_ADD_PARQUET_TEST(parquet_generator-test parquet_generator-test.cpp)
# target_compile_definitions(parquet_generator-test
# PUBLIC -DPARQUET_FILE_PATH="${PARQUET_FILE_PATH}")
