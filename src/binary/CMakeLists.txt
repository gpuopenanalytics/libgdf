
option(BINARY_OPERATION_VERSION "Select binary operation version" "V2")

if (${BINARY_OPERATION_VERSION} STREQUAL "V1")
    message("-- Binary Operation Version: V1")
    list(APPEND gdfs_source_files
         "${CMAKE_CURRENT_LIST_DIR}/common/types.cpp"
         "${CMAKE_CURRENT_LIST_DIR}/common/mediator.cu"
    )
endif()

if (${BINARY_OPERATION_VERSION} STREQUAL "V2")
    message("-- Binary Operation Version: V2 - NVRTC")
    list(APPEND gdfs_source_files
         "${CMAKE_CURRENT_LIST_DIR}/binary2/binary.cpp"
         "${CMAKE_CURRENT_LIST_DIR}/binary2/kernel_gdf_data.cpp"
         "${CMAKE_CURRENT_LIST_DIR}/binary2/kernel.cpp"
         "${CMAKE_CURRENT_LIST_DIR}/binary2/launcher.cpp"
         "${CMAKE_CURRENT_LIST_DIR}/binary2/operation.cpp"
         "${CMAKE_CURRENT_LIST_DIR}/binary2/traits.cpp"
         "${CMAKE_CURRENT_LIST_DIR}/binary2/type.cpp"
    )
endif()
