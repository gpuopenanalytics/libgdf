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

function(ConfigureTest  test_name  test_source)
    ## Add link directory
    link_directories(${CUDA_LIBRARY_DIR})
    link_directories(${CUDA_LIBRARY_STUBS_DIR})

    ## Add test
    add_executable(${test_name} ${test_source})
    add_test(NAME ${test_name} COMMAND ${test_name})

    ## Clear include header property
    set_target_properties(${test_name} PROPERTIES INCLUDE_DIRECTORIES "")
    set_target_properties(${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests/")

    ## Include headers
    target_include_directories(${test_name} PRIVATE "${GTEST_INCLUDE_DIRS}")
    target_include_directories(${test_name} PRIVATE "${CUDA_INCLUDE_DIRS}")
    target_include_directories(${test_name} PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_include_directories(${test_name} PRIVATE "${PROJECT_SOURCE_DIR}/src")

    ## Link libraries
    target_link_libraries(${test_name} ${GTEST_BOTH_LIBRARIES})
    target_link_libraries(${test_name} ${GTEST_LIBRARIES})
    target_link_libraries(${test_name} gdf)
    target_link_libraries(${test_name} ${CUDA_CUDA_LIB})
    target_link_libraries(${test_name} ${CUDA_NVRTC_LIB})
endfunction()


## BinaryOperationIntegrationTest
set(binop_source
    "${PROJECT_SOURCE_DIR}/src/tests/binary-operation/util/types.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/binary-operation-integration-test.cpp")

ConfigureTest("BinaryOperationIntegrationTest" "${binop_source}")

## BinaryOperationOperandsNullTest
set(binop_source
    "${PROJECT_SOURCE_DIR}/src/tests/binary-operation/util/types.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/binary-operation-operands-null-test.cpp")

ConfigureTest("BinaryOperationOperandsNullTest" "${binop_source}")
