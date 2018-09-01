
option(Benchmark "Benchmark" OFF)

if(Benchmark)
  message("-- Benchmark Active")
  include(GoogleBenchmark)
  add_subdirectory(src/bench)
endif()
