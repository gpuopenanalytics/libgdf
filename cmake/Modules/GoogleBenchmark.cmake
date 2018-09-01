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
