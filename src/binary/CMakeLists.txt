# cmake -DCMAKE_BUILD_TYPE=Release -DBINARY_OPERATION_VERSION:STRING=V1 ../../code/libgdf

if (NOT DEFINED BINARY_OPERATION_VERSION)
    list(APPEND gdfs_source_files "")
    return()
endif()


if (${BINARY_OPERATION_VERSION} STREQUAL "V1")
    message("BINARY_OPERATION_VERSION: V1 Selected")
    list(APPEND gdfs_source_files
         "${CMAKE_CURRENT_LIST_DIR}/common/types.cpp"
         "${CMAKE_CURRENT_LIST_DIR}/common/mediator.cu"
    )
endif()


if (${BINARY_OPERATION_VERSION} STREQUAL "V2")
    message("BINARY_OPERATION_VERSION: V2 Selected")
    list(APPEND gdfs_source_files
         "${CMAKE_CURRENT_LIST_DIR}/binary2/binary.cpp"
         "${CMAKE_CURRENT_LIST_DIR}/binary2/kernel.cpp"
         "${CMAKE_CURRENT_LIST_DIR}/binary2/launcher.cpp"
         "${CMAKE_CURRENT_LIST_DIR}/binary2/operation.cpp"
         "${CMAKE_CURRENT_LIST_DIR}/binary2/traits.cpp"
         "${CMAKE_CURRENT_LIST_DIR}/binary2/type.cpp"
    )
endif()
