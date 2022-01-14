cmake_minimum_required (VERSION 3.8.2)

project (mkldnn)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------- Cmake Options ----------------------------
# Reset the default install prefix
if( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
  message(STATUS "Setting default CMAKE_INSTALL_PREFIX path to ${CMAKE_BINARY_DIR}/install")
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE STRING "The path to use for make install" FORCE)
endif()

# Reset the default output dir
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib" CACHE PATH "Archive output dir.")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib" CACHE PATH "Library output dir.")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin" CACHE PATH "Executable/dll output dir.")

# Set build type
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  message(STATUS "CMAKE_BUILD_TYPE is unset, defaulting to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: None Debug Release ...")
endif()

# ---------------------------- C++11 Check ----------------------------

include(CheckCXXCompilerFlag)

CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)

# C++11 Flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# cxx11 abi setting
if(DEFINED USE_CXX11_ABI)
  if (USE_CXX11_ABI)
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
  else()
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
  endif()
endif()

# ----- The original OneDNN Lib -----
# Library: dnnl
message(STATUS "Build dnnl static lib")
set(DNNL_LIBRARY_TYPE "STATIC" CACHE STRING "")
option(DNNL_BUILD_EXAMPLES OFF)
option(DNNL_BUILD_TESTS OFF)
add_subdirectory(oneDNN)

install(TARGETS dnnl
  LIBRARY DESTINATION lib64
  ARCHIVE DESTINATION lib64
  PUBLIC_HEADER DESTINATION include)
