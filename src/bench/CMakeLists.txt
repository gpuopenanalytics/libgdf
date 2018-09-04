## Test name
set(GDF_APP_NAME  BinaryOperationsBench)
set(GDF_BENCH_DIR bench)

## Set sources
set(GDF_SOURCE_FILES
    "${CMAKE_SOURCE_DIR}/src/library/types.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/binary-operation-benchmark.cpp")

## Add link directory
link_directories(${CUDA_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_STUBS_DIR})
link_directories(${GOOGLE_BENCHMARK_LIBRARY_DIR})

## Add target
add_executable(${GDF_APP_NAME} ${GDF_SOURCE_FILES})

## Clear include header property
set_target_properties(${GDF_APP_NAME}
                      PROPERTIES
                      INCLUDE_DIRECTORIES "")
set_target_properties(${GDF_APP_NAME}
                      PROPERTIES
                      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${GDF_BENCH_DIR}")

## Include headers
target_include_directories(${GDF_APP_NAME} PRIVATE "${CUDA_INCLUDE_DIRS}")
target_include_directories(${GDF_APP_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_include_directories(${GDF_APP_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_include_directories(${GDF_APP_NAME} PRIVATE "${GOOGLE_BENCHMARK_INCLUDE_DIR}")

## Link libraries
target_link_libraries(${GDF_APP_NAME} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(${GDF_APP_NAME} ${GOOGLE_BENCHMARK_LIB})
target_link_libraries(${GDF_APP_NAME} ${GOOGLE_BENCHMARK_MAIN_LIB})
target_link_libraries(${GDF_APP_NAME} gdf)
target_link_libraries(${GDF_APP_NAME} ${CUDA_CUDA_LIB})
target_link_libraries(${GDF_APP_NAME} ${CUDA_NVRTC_LIB})

message("-- benchmark for binary operations")
message(" * Execute help: ${GDF_BENCH_DIR}/${GDF_APP_NAME} --help")
message(" * Execute benchmark: ${GDF_BENCH_DIR}/${GDF_APP_NAME} --benchmark_counters_tabular=true")
