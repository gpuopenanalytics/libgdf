
set(BINARY_OPERATION_VERSION "V2" CACHE STRING "Select binary operation version")

if (${BINARY_OPERATION_VERSION} STREQUAL "V1")
    message("-- Binary Operation Version: V1")
    list(APPEND source_files
         "${CMAKE_CURRENT_LIST_DIR}/common/types.cpp"
         "${CMAKE_CURRENT_LIST_DIR}/common/mediator.cu"
    )
endif()

if (${BINARY_OPERATION_VERSION} STREQUAL "V2")
    message("-- Binary Operation Version: V2 - NVRTC")
    set(source_files
        src/binary/binary2/binary.cpp
        src/binary/binary2/kernel_gdf_data.cpp
        src/binary/binary2/kernel.cpp
        src/binary/binary2/launcher.cpp
        src/binary/binary2/operation.cpp
        src/binary/binary2/traits.cpp
        src/binary/binary2/type.cpp
    )
endif()
