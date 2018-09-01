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