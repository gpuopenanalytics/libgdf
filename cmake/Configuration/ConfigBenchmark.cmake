## Build Benchmark
## By default is disabled.
## Add -DBENCHMARK:BOOL=ON as cmake parameter to create the benchmark.

option(BENCHMARK "Benchmark" OFF)

if(BENCHMARK)
    message(STATUS "Benchmark is enabled")
    include(GoogleBenchmark)
    add_subdirectory(src/bench)
endif()
