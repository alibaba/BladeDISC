list(APPEND TAO_BRIDGE_HEADERS
    "common.h"
    "dumper_common.h"
    "executable.h"
    "tao_util.h"
    "errors.h"
    "cuda_utils.h"
)

list(APPEND TAO_BRIDGE_SOURCES
    "common.cc"
    "dumper_common.cc"
    "version.cc"
    "executable.cc"
    "tao_util.cc"
    "errors.cc"
    "cuda_utils.cc"
)

list(APPEND PROTO_FILES
    "tao_compiler_input.proto"
    "tao_compilation_result.proto"
    "xla_activity.proto"
)

list(APPEND TAO_BRIDGE_TESTS
    "tao_util_test.cc"
    "errors_test.cc"
    "common_test.cc"
    "dumper_common_test.cc"
    "cuda_utils_test.cc"
)

# generate protobuf file
foreach(file ${PROTO_FILES})
    message("Generating proto: ${file}")
    EXECUTE_PROCESS(
        COMMAND ${Protobuf_PROTOC_EXECUTABLE} --cpp_out=. ${file}
        WORKING_DIRECTORY ${tao_ops_SOURCE_DIR}/tao_bridge
        RESULT_VARIABLE CMD_RESULT
        OUTPUT_VARIABLE STDOUT
        ERROR_VARIABLE STDERR
    )
    if (NOT CMD_RESULT EQUAL "0")
        message(FATAL_ERROR "Generating proto failed: \n${STDOUT}\n${STDERR}")
    endif()
endforeach()

file(GLOB PROTO_SOURCES "*.pb.cc")
file(GLOB PROTO_HEADERS "*.pb.h")

if (Tensorflow_USE_PB3)
    MESSAGE("Renaming namespace from google::protobuf to google::protobuf3.")
    EXECUTE_PROCESS(
        COMMAND sed -i "s#::protobuf::#::protobuf3::#g; s#namespace protobuf #namespace protobuf3 #g" ${PROTO_SOURCES} ${PROTO_HEADERS}
        WORKING_DIRECTORY .
    )
endif()


add_library(tao_bridge OBJECT
    ${TAO_BRIDGE_SOURCES}
    ${PROTO_SOURCES}
)
tao_export_build_env_var()
configure_file(version.h.in version.h @ONLY)
target_include_directories(tao_bridge PRIVATE ${CMAKE_BINARY_DIR})
include_directories("ral")

if(${PLATFORM_ALIBABA})
  include_directories(../../platform_alibaba)
  add_subdirectory(../../platform_alibaba tao_launch_op)
endif()

add_subdirectory(mlir)
add_subdirectory(kernels)
add_subdirectory(ops)
add_subdirectory(passes)
add_subdirectory(ral)
add_subdirectory(tf)

tao_cc_test(
  NAME tao_bridge_tests
  SRCS ${TAO_BRIDGE_TESTS}
  EXTRALIBS dl
)

