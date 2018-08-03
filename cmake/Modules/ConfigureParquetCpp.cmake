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

# Download and unpack ParquetCpp at configure time
configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/ParquetCpp.CMakeLists.txt.cmake ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/parquetcpp-download/CMakeLists.txt)

execute_process(
    COMMAND ${CMAKE_COMMAND} -F "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/parquetcpp-download/
)

if(result)
    message(FATAL_ERROR "CMake step for ParquetCpp failed: ${result}")
endif()

# Transitive dependencies
set(ARROW_TRANSITIVE_DEPENDENCIES_PREFIX ${ARROW_DOWNLOAD_BINARY_DIR}/arrow-prefix/src/arrow-build)
set(BROTLI_TRANSITIVE_DEPENDENCY_PREFIX ${ARROW_TRANSITIVE_DEPENDENCIES_PREFIX}/brotli_ep/src/brotli_ep-install/lib/x86_64-linux-gnu)
set(BROTLI_STATIC_LIB_ENC ${BROTLI_TRANSITIVE_DEPENDENCY_PREFIX}/libbrotlienc.a)
set(BROTLI_STATIC_LIB_DEC ${BROTLI_TRANSITIVE_DEPENDENCY_PREFIX}/libbrotlidec.a)
set(BROTLI_STATIC_LIB_COMMON ${BROTLI_TRANSITIVE_DEPENDENCY_PREFIX}/libbrotlicommon.a)
set(SNAPPY_STATIC_LIB ${ARROW_TRANSITIVE_DEPENDENCIES_PREFIX}/snappy_ep/src/snappy_ep-install/lib/libsnappy.a)
set(ZLIB_STATIC_LIB ${ARROW_TRANSITIVE_DEPENDENCIES_PREFIX}/zlib_ep/src/zlib_ep-install/lib/libz.a)
set(ARROW_HOME ${ARROW_ROOT})

set(ENV{BROTLI_STATIC_LIB_ENC} ${BROTLI_STATIC_LIB_ENC})
set(ENV{BROTLI_STATIC_LIB_DEC} ${BROTLI_STATIC_LIB_DEC})
set(ENV{BROTLI_STATIC_LIB_COMMON} ${BROTLI_STATIC_LIB_COMMON})
set(ENV{SNAPPY_STATIC_LIB} ${SNAPPY_STATIC_LIB})
set(ENV{ZLIB_STATIC_LIB} ${ZLIB_STATIC_LIB})
set(ENV{ARROW_HOME} ${ARROW_HOME})

execute_process(
    COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/parquetcpp-download)

if(result)
    message(FATAL_ERROR "Build step for ParquetCpp failed: ${result}")
endif()

# Add transitive dependency: Thrift
set(THRIFT_ROOT ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/parquetcpp-build/thrift_ep/src/thrift_ep-install)

# Locate ParquetCpp package
set(PARQUETCPP_ROOT ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/parquetcpp-install)
set(PARQUETCPP_BINARY_DIR ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/parquetcpp-build)
set(PARQUETCPP_SOURCE_DIR ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/parquetcpp-src)

# Dependency interfaces
find_package(Boost REQUIRED COMPONENTS regex)

add_library(Apache::Thrift INTERFACE IMPORTED)
set_target_properties(Apache::Thrift
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${THRIFT_ROOT}/include)
set_target_properties(Apache::Thrift
    PROPERTIES INTERFACE_LINK_LIBRARIES ${THRIFT_ROOT}/lib/libthrift.a)

add_library(Apache::Arrow INTERFACE IMPORTED)
set_target_properties(Apache::Arrow
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${ARROW_ROOT}/include)
set_target_properties(Apache::Arrow
    PROPERTIES INTERFACE_LINK_LIBRARIES "${ARROW_ROOT}/lib/libarrow.a;${BROTLI_STATIC_LIB_ENC};${BROTLI_STATIC_LIB_DEC};${BROTLI_STATIC_LIB_COMMON};${SNAPPY_STATIC_LIB};${ZLIB_STATIC_LIB}")

add_library(Apache::ParquetCpp INTERFACE IMPORTED)
set_target_properties(Apache::ParquetCpp
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
        "${PARQUETCPP_ROOT}/include;${PARQUETCPP_BINARY_DIR}/src;${PARQUETCPP_SOURCE_DIR}/src")
set_target_properties(Apache::ParquetCpp
    PROPERTIES INTERFACE_LINK_LIBRARIES "${PARQUETCPP_ROOT}/lib/libparquet.a;Apache::Arrow;Apache::Thrift;Boost::regex")
