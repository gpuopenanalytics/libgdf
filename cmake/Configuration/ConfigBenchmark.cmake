## Build Benchmark
## By default is disabled.
## Add -DBenchmark=ON as cmake parameter to create the benchmark.

option(Benchmark "Benchmark" OFF)

if(Benchmark)
  message("-- Benchmark Active")
  include(GoogleBenchmark)
  add_subdirectory(src/bench)
endif()
