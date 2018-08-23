## Test name
set(GDF_TEST_NAME BinaryOperations)


## Set sources
set(GDF_SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/library/types.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/integration/binary-operation-test.cpp")


## Add test
add_executable(${GDF_TEST_NAME} ${GDF_SOURCE_FILES})
add_test(NAME ${GDF_TEST_NAME} COMMAND ${GDF_TEST_NAME})


## Clear include header property
set_target_properties(${GDF_TEST_NAME} PROPERTIES INCLUDE_DIRECTORIES "")
set_target_properties(${GDF_TEST_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests/")


## Include headers
target_include_directories(${GDF_TEST_NAME} PUBLIC "${GTEST_INCLUDE_DIRS}")
target_include_directories(${GDF_TEST_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(${GDF_TEST_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/include")


## Link libraries
target_link_libraries(${GDF_TEST_NAME} ${GTEST_BOTH_LIBRARIES})
target_link_libraries(${GDF_TEST_NAME} gdf)
target_link_libraries(${GDF_TEST_NAME} cuda)
target_link_libraries(${GDF_TEST_NAME} nvrtc)
