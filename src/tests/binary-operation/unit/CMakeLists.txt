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

## Test name
set(GDF_TEST_NAME BinaryOperationsVerifyInputTest)


## Set sources
set(GDF_SOURCE_FILES
    "${PROJECT_SOURCE_DIR}/src/tests/binary-operation/util/types.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/binop-verify-input-test.cpp")


## Add link directory
link_directories(${CUDA_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_STUBS_DIR})


## Add test
add_executable(${GDF_TEST_NAME} ${GDF_SOURCE_FILES})
add_test(NAME ${GDF_TEST_NAME} COMMAND ${GDF_TEST_NAME})


## Clear include header property
set_target_properties(${GDF_TEST_NAME} PROPERTIES INCLUDE_DIRECTORIES "")
set_target_properties(${GDF_TEST_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests/")


## Include headers
target_include_directories(${GDF_TEST_NAME} PRIVATE "${GTEST_INCLUDE_DIRS}")
target_include_directories(${GDF_TEST_NAME} PRIVATE "${CUDA_INCLUDE_DIRS}")
target_include_directories(${GDF_TEST_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_include_directories(${GDF_TEST_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/src")


## Link libraries
target_link_libraries(${GDF_TEST_NAME} ${GTEST_BOTH_LIBRARIES})
target_link_libraries(${GDF_TEST_NAME} gdf)
target_link_libraries(${GDF_TEST_NAME} ${CUDA_CUDA_LIB})
target_link_libraries(${GDF_TEST_NAME} ${CUDA_NVRTC_LIB})
