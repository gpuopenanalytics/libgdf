
if(GDF_BENCHMARK)

 include(ExternalProject)

 ExternalProject_Add(benchmark_ep
 	CMAKE_ARGS
 		-DCMAKE_BUILD_TYPE=RELEASE
 		-DCMAKE_INSTALL_PREFIX=build
 		-DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON
 	GIT_REPOSITORY https://github.com/google/benchmark.git
     GIT_TAG v1.4.1
 	UPDATE_COMMAND ""
 )
 ExternalProject_Get_property(benchmark_ep BINARY_DIR)
 set(BENCHMARK_ROOT ${BINARY_DIR}/build)

 file(MAKE_DIRECTORY ${BENCHMARK_ROOT}/include)
 file(MAKE_DIRECTORY ${BENCHMARK_ROOT}/lib)

 add_library(Google::Benchmark INTERFACE IMPORTED)
 add_dependencies(Google::Benchmark benchmark_ep)
 set_target_properties(Google::Benchmark
     PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${BENCHMARK_ROOT}/include)
 set_target_properties(Google::Benchmark
     PROPERTIES INTERFACE_LINK_LIBRARIES ${BENCHMARK_ROOT}/lib/libbenchmark.a)

 add_library(Google::Benchmark::Main INTERFACE IMPORTED)
 set_target_properties(Google::Benchmark::Main
     PROPERTIES INTERFACE_LINK_LIBRARIES ${BENCHMARK_ROOT}/lib/libbenchmark_main.a)


function(GDF_ADD_BENCHMARK TARGET)
    list(REMOVE_AT ARGV 0)
    cuda_add_executable(${TARGET} ${ARGV})
    target_link_libraries(${TARGET}
        Google::Benchmark Google::Benchmark::Main gdf)
endfunction()

endif()
 

 add_subdirectory(replace) 