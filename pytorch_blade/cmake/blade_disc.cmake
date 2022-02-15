if (DEFINED ENV{USE_BLADE_DISC_PRE_BUILD})
  # This flag was introduced to speedup torch_blade building for developers.
  # If USE_BLADE_DISC_PRE_BUILD is ON, it means the blade_disc has been built previously,
  # and cmake would skip blade_disc compilation to save time.
  option(USE_BLADE_DISC_PRE_BUILD "Use the blade_disc libraries that had been built previously" $ENV{USE_BLADE_DISC_PRE_BUILD})
else()
  option(USE_BLADE_DISC_PRE_BUILD "Use the blade_disc libraries that had been built previously" OFF)
endif()

if (USE_BLADE_DISC_PRE_BUILD)
  message(STATUS "Use the blade_disc libraries that had been built previously")
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

list(APPEND BLADE_DISC_CONFIG --py_ver ${PYTHON_EXECUTABLE})
if (NOT TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT)
    list(APPEND BLADE_DISC_CONFIG --cpu_only)
endif (NOT TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT)

if (TORCH_BLADE_USE_ROCM)
    list(APPEND BLADE_DISC_CONFIG --dcu)
endif (TORCH_BLADE_USE_ROCM)

if (DEFINED TORCH_BLADE_USE_CXX11_ABI AND TORCH_BLADE_USE_CXX11_ABI)
    list(APPEND BLADE_DISC_CONFIG --ral_cxx11_abi)
endif()

message(STATUS "Use script ${CMAKE_CURRENT_SOURCE_DIR}/ci_build/build_blade_disc.sh")
execute_process(
  COMMAND
  "bash" ${CMAKE_CURRENT_SOURCE_DIR}/ci_build/build_blade_disc.sh
  ${BLADE_DISC_CONFIG}

  WORKING_DIRECTORY
  ${CMAKE_CURRENT_SOURCE_DIR}/..

  RESULT_VARIABLE
  BUILD_DISC_RESULT
)
if (NOT BUILD_DISC_RESULT EQUAL "0")
    message(FATAL_ERROR "Build BladeDISC failed, error code: ${BUILD_DISC_RESULT}")
endif()

endif()

SET(BLADE_DISC_INSTALL_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
SET(RAL_LIBNAME libral_base_context.so)
SET(MLIR_BUILDER_LIBNAME mlir_disc_builder.so)

add_custom_target(
  blade_disc
  COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/../tf_community/bazel-bin/tensorflow/compiler/mlir/xla/ral/${RAL_LIBNAME} ${BLADE_DISC_INSTALL_DIR}
  COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/../tf_community/bazel-bin/tensorflow/compiler/mlir/disc/disc_compiler_main ${BLADE_DISC_INSTALL_DIR}
  COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/../tf_community/bazel-bin/tensorflow/compiler/mlir/disc/${MLIR_BUILDER_LIBNAME} ${BLADE_DISC_INSTALL_DIR}

  BYPRODUCTS ${BLADE_DISC_INSTALL_DIR}/${RAL_LIBNAME};${BLADE_DISC_INSTALL_DIR}/disc_compiler_main;${BLADE_DISC_INSTALL_DIR}/${MLIR_BUILDER_LIBNAME}

  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
install(PROGRAMS ${BLADE_DISC_INSTALL_DIR}/${RAL_LIBNAME};${BLADE_DISC_INSTALL_DIR}/${MLIR_BUILDER_LIBNAME} TYPE LIB)
