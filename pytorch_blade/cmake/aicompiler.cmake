if (DEFINED ENV{USE_AICOMPILER_PRE_BUILD})
  # This flag was introduced to speedup torch_blade building for developers.
  # If USE_AICOMPILER_PRE_BUILD is ON, it means the aicompiler has been built previously,
  # and cmake would skip aicompiler compilation to save time.
  option(USE_AICOMPILER_PRE_BUILD "Use the aicompiler libraries that had been built previously" $ENV{USE_AICOMPILER_PRE_BUILD})
else()
  option(USE_AICOMPILER_PRE_BUILD "Use the aicompiler libraries that had been built previously" OFF)
endif()

if (USE_AICOMPILER_PRE_BUILD)
  message(STATUS "Use the aicompiler libraries that had been built previously")
else()
execute_process(
  COMMAND
    "which" "gcc"
  OUTPUT_VARIABLE
  DEFAULT_GCC_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (DEFINED TORCH_BLADE_CUDA_VERSION)
   set(ENV{TAO_DOCKER_CUDA_VERSION} ${TORCH_BLADE_CUDA_VERSION})
endif()
message(STATUS "TAO_DOCKER_CUDA_VERSION=$ENV{TAO_DOCKER_CUDA_VERSION}")

# export GCC_HOST_COMPILER_PATH as default gcc
set(ENV{GCC_HOST_COMPILER_PATH} ${DEFAULT_GCC_PATH})
message(STATUS "Build disc with ${DEFAULT_GCC_PATH}")

list(APPEND AICOMPILER_CONFIG --py_ver ${PYTHON_EXECUTABLE})
if (NOT TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT)
    list(APPEND AICOMPILER_CONFIG --cpu_only)
endif (NOT TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT)

if (TORCH_BLADE_USE_ROCM)
    list(APPEND AICOMPILER_CONFIG --dcu)
endif (TORCH_BLADE_USE_ROCM)

if (DEFINED TORCH_BLADE_USE_CXX11_ABI AND TORCH_BLADE_USE_CXX11_ABI)
    list(APPEND AICOMPILER_CONFIG --ral_cxx11_abi)
endif()

message(STATUS "Use script ${CMAKE_CURRENT_SOURCE_DIR}/ci_build/build_aicompiler.sh")
execute_process(
  COMMAND
  "bash" ${CMAKE_CURRENT_SOURCE_DIR}/ci_build/build_aicompiler.sh
  ${AICOMPILER_CONFIG}

  WORKING_DIRECTORY
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)
endif()

SET(AICOMPILER_INSTALL_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
SET(RAL_LIBNAME libral_base_context.so)
SET(MLIR_BUILDER_LIBNAME mlir_disc_builder.so)

add_custom_target(
  aicompiler
  COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/../tf_community/bazel-bin/tensorflow/compiler/mlir/xla/ral/${RAL_LIBNAME} ${AICOMPILER_INSTALL_DIR}
  COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/../tf_community/bazel-bin/tensorflow/compiler/mlir/disc/disc_compiler_main ${AICOMPILER_INSTALL_DIR}
  COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/../tf_community/bazel-bin/tensorflow/compiler/mlir/disc/${MLIR_BUILDER_LIBNAME} ${AICOMPILER_INSTALL_DIR}

  BYPRODUCTS ${AICOMPILER_INSTALL_DIR}/${RAL_LIBNAME};${AICOMPILER_INSTALL_DIR}/disc_compiler_main;${AICOMPILER_INSTALL_DIR}/${MLIR_BUILDER_LIBNAME}

  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
install(PROGRAMS ${AICOMPILER_INSTALL_DIR}/${RAL_LIBNAME};${AICOMPILER_INSTALL_DIR}/${MLIR_BUILDER_LIBNAME} TYPE LIB)
