cmake_minimum_required(VERSION 3.0)
project(mhlo_builder)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

file(GLOB_RECURSE MHLO_BUILDER_SRCS csrc/*.cpp)

add_library(mhlo_builder STATIC ${MHLO_BUILDER_SRCS})
SET(MLIR_DISC_BAZEL_BIN_PATH ${PROJECT_SOURCE_DIR}/../tf_community/bazel-bin/tensorflow/compiler/mlir/disc)
SET(MLIR_DISC_BUILDER_LIBNAME ${MLIR_DISC_BAZEL_BIN_PATH}/mlir_disc_builder.so)
SET(MLIR_DISC_INCLUDE ${MLIR_DISC_BAZEL_BIN_PATH}/mlir_disc_include)

SET(MHLO_INCLUDE ${MLIR_DISC_INCLUDE}/tensorflow/compiler/mlir/hlo/include)
SET(LLVM_INCLUDE ${MLIR_DISC_INCLUDE}/external/llvm-project/llvm/include)
SET(MLIR_INCLUDE ${MLIR_DISC_INCLUDE}/external/llvm-project/mlir/include)

set(MHLO_BUILDER_INCS ${PROJECT_SOURCE_DIR}/csrc ${MLIR_DISC_INCLUDE} ${MHLO_INCLUDE} ${LLVM_INCLUDE} ${MLIR_INCLUDE})
set_property(
  TARGET mhlo_builder
  APPEND PROPERTY
  LINK_FLAGS "-Wl,--exclude-libs,ALL")

target_include_directories(mhlo_builder PUBLIC
  ${MHLO_BUILDER_INCS}
)

target_link_libraries(mhlo_builder PUBLIC
  ${LLVM_STATIC_LIBRARIES}
  ${MLIR_DISC_BUILDER_LIBNAME}
)

if (DEFINED MHLO_USE_CXX11_ABI)
  if (MHLO_USE_CXX11_ABI)
      target_compile_definitions(mhlo_builder PUBLIC -D_GLIBCXX_USE_CXX11_ABI=1)
  else()
      target_compile_definitions(mhlo_builder PUBLIC -D_GLIBCXX_USE_CXX11_ABI=0)
  endif()
endif()

