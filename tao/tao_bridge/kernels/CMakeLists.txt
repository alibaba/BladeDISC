list(APPEND KERNELS_HEADERS
    "tao_compilation_cache.h"
    "tao_compilation_info_collector.h"
    "profiling.h"
    "tao_profiling_guided_compilation.h"
    "process.h"
    "disc_launch.h"
    "launch_base.h"
    "platform_info.h"
)

list(APPEND KERNELS_SOURCES
    "tao_compilation_cache.cc"
    "tao_compilation_info_collector.cc"
    "profiling.cc"
    "tao_profiling_guided_compilation.cc"
    "process.cc"
    "disc_launch.cc"
    "launch_base.cc"
    "platform_info.cc"
)

add_library(kernels OBJECT ${KERNELS_SOURCES})
target_include_directories(kernels PRIVATE ${CMAKE_BINARY_DIR})