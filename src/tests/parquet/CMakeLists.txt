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




function(GDF_ADD_PARQUET_TEST TARGET)
    message(STATUS "${TARGET} will link against: gdf parquet")
    list(REMOVE_AT ARGV 0)
    cuda_add_executable(${TARGET} ${ARGV})
    target_include_directories(${TARGET} PUBLIC ${CMAKE_SOURCE_DIR}/src/parquet)
    target_link_libraries(${TARGET} gmock_main gmock GTest::GTest gdf-parquet)
    get_property(ARGN TARGET ${TARGET} PROPERTY SOURCES)
    # gtest_add_tests(${TARGET} "" ${ARGN})
endfunction()

add_subdirectory(file_reader)
add_subdirectory(gdf_column)
