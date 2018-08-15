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

find_package(Boost REQUIRED COMPONENTS filesystem)

set(PARQUET_FILE_PATH
    ${CMAKE_SOURCE_DIR}/src/tests/parquet/file_reader/reader-test.parquet)

GDF_ADD_PARQUET_TEST(file_reader-test
    file_reader-test.cpp
    single_column_file-test.cpp
    api-test.cpp
    )

target_compile_definitions(file_reader-test
    PUBLIC -DPARQUET_FILE_PATH="${PARQUET_FILE_PATH}")
target_link_libraries(file_reader-test Boost::filesystem)

GDF_ADD_PARQUET_BENCHMARK(file_reader-benchmark file_reader-benchmark.cpp)

if (GDF_BENCHMARK)
target_compile_definitions(file_reader-benchmark
    PUBLIC -DPARQUET_FILE_PATH="${PARQUET_FILE_PATH}")
endif()
