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

# Download and unpack google-benchmark at configure time
configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/GoogleBenchmark.CMakeLists.txt.cmake
               ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/google-benchmark/download/CMakeLists.txt)

execute_process(
    COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/google-benchmark/download
)

if(result)
    message(FATAL_ERROR "CMake step for google benchmark failed: ${result}")
endif()


execute_process(
    COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/google-benchmark/download
)

if(result)
    message(FATAL_ERROR "Build step for google benchmark failed: ${result}")
endif()


set(GOOGLE_BENCHMARK_LIB         "benchmark")
set(GOOGLE_BENCHMARK_MAIN_LIB    "benchmark_main")
set(GOOGLE_BENCHMARK_DIR         "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/google-benchmark/install")
set(GOOGLE_BENCHMARK_INCLUDE_DIR "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/google-benchmark/install/include")
set(GOOGLE_BENCHMARK_LIBRARY_DIR "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/google-benchmark/install/lib")
