cmake_minimum_required(VERSION 3.0)
project(torch_blade)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(BUILD_SHARED_LIBS "Build libtorch_blade as shared library" ON)
option(TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT "Build libtorch_blade with cuda support" ON)
option(TORCH_BLADE_BUILD_MLIR_SUPPORT "Build libtorch_blade with mlir support" ON)
option(TORCH_BLADE_BUILD_PYTHON_SUPPORT "Build torch_blade with python support" ON)
option(TORCH_BLADE_PLATFORM_ALIBABA "Build torch_blade inside alibaba" OFF)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}") # CMake 3.9+
endif()

# Always build the libraries with -fPIC. This is because the Python
# extension is always a shared library, and it links to
# libtorch_blade.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (DEFINED TORCH_BLADE_USE_CXX11_ABI)
  if (TORCH_BLADE_USE_CXX11_ABI)
      set(MHLO_USE_CXX11_ABI ON)
      add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
  else()
      set(MHLO_USE_CXX11_ABI OFF)
      add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
  endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# cmake utils
include(cmake/utils.cmake)
# PYTORCH_DIR
resolve_env(PYTORCH_DIR)

set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations ${CMAKE_CXX_FLAGS}")

# GCC_PATH
if(DEFINED ENV{GCC_PATH})
  set(GCC_PATH $ENV{GCC_PATH})
  message(STATUS "SET GCC_PATH=${GCC_PATH}")
  if(CMAKE_BUILD_RPATH)
	  set(CMAKE_BUILD_RPATH "${GCC_PATH}/lib64:${CMAKE_BUILD_RPATH}")
  else()
	  set(CMAKE_BUILD_RPATH "${GCC_PATH}/lib64")
  endif()
endif()

if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
  set(_portable_origin_rpath "@loader_path")
else()
  set(_portable_origin_rpath "$ORIGIN")
endif(APPLE)

if(CMAKE_BUILD_RPATH)
  set(CMAKE_BUILD_RPATH "${_portable_origin_rpath}:${CMAKE_BUILD_RPATH}")
else()
  set(CMAKE_BUILD_RPATH "${_portable_origin_rpath}")
endif()

set(CMAKE_INSTALL_RPATH "${CMAKE_BUILD_RPATH}")
set(CMAKE_SKIP_BUILD_RPATH ON)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

file(GLOB_RECURSE TORCH_BLADE_TEST_SRCS src/*test.cpp)
file(GLOB TORCH_BLADE_PYTHON_SRCS src/pybind*.cpp src/compiler/jit/pybind*.cpp src/compiler/jit/torch/pybind*.cpp)
file(GLOB TORCH_BLADE_PYTHON_HEADERS src/pybind*.h src/compiler/jit/pybind*.h src/compiler/jit/torch/pybind*.h)
file(GLOB_RECURSE TORCH_BLADE_SRCS src/common_utils/*.cpp src/compiler/jit/*.cpp src/compiler/jit/torch/*.cpp)
file(GLOB_RECURSE TORCH_BLADE_HEADERS src/common_utils/*.h src/compiler/jit/*.h)

if (TORCH_BLADE_BUILD_MLIR_SUPPORT)
  file(GLOB_RECURSE TORCH_BLADE_MLIR_PYTHON_SRCS src/compiler/mlir/*pybind*.cpp)
  file(GLOB_RECURSE TORCH_BLADE_MLIR_PYTHON_HEADERS src/compiler/mlir/*pybind*.h)
  file(GLOB_RECURSE TORCH_BLADE_MLIR_SRCS src/compiler/mlir/*.cpp)
  file(GLOB_RECURSE TORCH_BLADE_MLIR_HEADERS src/compiler/mlir/*.h)
  list(APPEND TORCH_BLADE_PYTHON_SRCS ${TORCH_BLADE_MLIR_PYTHON_SRCS})
  list(APPEND TORCH_BLADE_PYTHON_HEADERS ${TORCH_BLADE_MLIR_PYTHON_HEADERS})
  list(APPEND TORCH_BLADE_SRCS ${TORCH_BLADE_MLIR_SRCS})
  list(APPEND TORCH_BLADE_HEADERS ${TORCH_BLADE_MLIR_HEADERS})
endif (TORCH_BLADE_BUILD_MLIR_SUPPORT)


get_filename_component(ONNX_SRC "src/compiler/jit/torch/onnx.cpp" ABSOLUTE)
get_filename_component(ONNX_HEADER "src/compiler/jit/torch/onnx.h" ABSOLUTE)
list(APPEND TORCH_BLADE_PYTHON_SRCS ${ONNX_SRC})
list(APPEND TORCH_BLADE_PYTHON_HEADERS ${ONNX_HEADER})

exclude(TORCH_BLADE_SRCS "${TORCH_BLADE_SRCS}" "${TORCH_BLADE_PYTHON_SRCS}" "${TORCH_BLADE_TEST_SRCS}")
exclude(TORCH_BLADE_HEADERS "${TORCH_BLADE_HEADERS}" "${TORCH_BLADE_PYTHON_HEADERS}")

# libtorch_blade, no dependency to Python
add_library(torch_blade ${TORCH_BLADE_SRCS} ${TORCH_BLADE_HEADERS})

if(TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT)
  resolve_env(TORCH_BLADE_CUDA_VERSION)
  resolve_env(CUDA_HOME)
  target_include_directories(torch_blade PUBLIC ${CUDA_HOME}/include)
  set(TORCH_CUDA ON)
endif (TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT)

if (TORCH_BLADE_BUILD_MLIR_SUPPORT)
  include(cmake/rocm.cmake)
  # build RAL_LIBNAME and DISC_COMPILER
  include(cmake/blade_disc.cmake)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../mhlo_builder ${CMAKE_CURRENT_BINARY_DIR}/mhlo_builder EXCLUDE_FROM_ALL)
  SET(RAL_LIBRARY ${BLADE_DISC_INSTALL_DIR}/${RAL_LIBNAME} mhlo_builder)
endif (TORCH_BLADE_BUILD_MLIR_SUPPORT)

if (TORCH_BLADE_BUILD_PYTHON_SUPPORT)
  # python binding
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/pybind11")
  include(cmake/pybind11/pybind11Tools.cmake)
  pybind11_add_module(_torch_blade SHARED
   ${TORCH_BLADE_PYTHON_SRCS}
   ${TORCH_BLADE_PYTHON_HEADERS}
  )
endif (TORCH_BLADE_BUILD_PYTHON_SUPPORT)

if (TORCH_BLADE_PLATFORM_ALIBABA)
  if (EXISTS ${CMAKE_SOURCE_DIR}/cmake/inside_alibaba.cmake)
    include(${CMAKE_SOURCE_DIR}/cmake/inside_alibaba.cmake)
  endif ()
endif (TORCH_BLADE_PLATFORM_ALIBABA)

# Common include path.
target_include_directories(torch_blade PUBLIC
  src/
)

find_package(Torch REQUIRED HINTS ${CMAKE_SOURCE_DIR}/cmake)
target_link_libraries(torch_blade PUBLIC "${TORCH_LIBRARIES}")
target_link_libraries(torch_blade PUBLIC
  ${RAL_LIBRARY}
)

# get torch version into version.h
resolve_env(PYTORCH_VERSION_STRING)
resolve_env(PYTORCH_MAJOR_VERSION)
resolve_env(PYTORCH_MINOR_VERSION)
target_compile_definitions(torch_blade PUBLIC
   -DPYTORCH_VERSION_STRING=${PYTORCH_VERSION_STRING}
   -DPYTORCH_MAJOR_VERSION=${PYTORCH_MAJOR_VERSION}
   -DPYTORCH_MINOR_VERSION=${PYTORCH_MINOR_VERSION}
)

if (TORCH_BLADE_BUILD_MLIR_SUPPORT)
  add_dependencies(torch_blade blade_disc)
endif (TORCH_BLADE_BUILD_MLIR_SUPPORT)
if (TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT)
  target_compile_definitions(torch_blade PUBLIC -DTORCH_BLADE_BUILD_WITH_CUDA)
endif (TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT)

add_subdirectory(third_party/googletest ${CMAKE_CURRENT_BINARY_DIR}/googletest EXCLUDE_FROM_ALL)
enable_testing()
add_gtest(shape_type_test
  SRCS src/compiler/jit/shape_type_test.cpp
  LIBS torch_blade
)

if (TORCH_BLADE_BUILD_MLIR_SUPPORT)
  add_gtest(mhlo_converter_register_test
    SRCS src/compiler/mlir/converters/mhlo_converter_register_test.cpp
    LIBS torch_blade
  )
endif (TORCH_BLADE_BUILD_MLIR_SUPPORT)

if (TORCH_BLADE_BUILD_PYTHON_SUPPORT)
  # python binding
  if (TORCH_BLADE_BUILD_MLIR_SUPPORT)
    target_compile_definitions(_torch_blade PRIVATE -DTORCH_BLADE_BUILD_MLIR)
  endif (TORCH_BLADE_BUILD_MLIR_SUPPORT)

  target_include_directories(_torch_blade PUBLIC
    ${PYTORCH_DIR}/include
    src/
  )

  target_link_directories(_torch_blade PUBLIC
    ${PYTORCH_DIR}/lib
  )

  target_link_libraries(_torch_blade PUBLIC
    torch_python torch_blade)
  set_target_properties(_torch_blade PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                           SUFFIX "${PYTHON_MODULE_EXTENSION}")
endif (TORCH_BLADE_BUILD_PYTHON_SUPPORT)

install(TARGETS torch_blade LIBRARY PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/version.txt TYPE DOC)
set(CPACK_PACKAGE_NAME "torch_blade")
set(CPACK_RPM_PACKAGE_GROUP "PAI-Blade")
set(CPACK_PACKAGE_VENDOR "AlibabaGroup")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "PAI-Blade")
set(CPACK_GENERATOR "TGZ")
include(CPack)
