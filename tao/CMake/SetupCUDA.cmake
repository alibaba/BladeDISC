################################
# Sanity Checks
################################

# Rare case of two flags being incompatible
# if (DEFINED CMAKE_SKIP_BUILD_RPATH AND DEFINED CUDA_LINK_WITH_NVCC)
#   if (NOT CMAKE_SKIP_BUILD_RPATH AND CUDA_LINK_WITH_NVCC)
#     message( FATAL_ERROR
#       "CMAKE_SKIP_BUILD_RPATH (FALSE) and CUDA_LINK_WITH_NVCC (TRUE) "
#       "are incompatible when linking explicit shared libraries. Set "
#       "CMAKE_SKIP_BUILD_RPATH to TRUE.")
#   endif()
# endif()

# CUDA_HOST_COMPILER was changed in 3.9.0 to CMAKE_CUDA_HOST_COMPILER and
# needs to be set prior to enabling the CUDA language
get_property(_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.9.0" )
    if ( NOT CMAKE_CUDA_HOST_COMPILER )
        if("CUDA" IN_LIST _languages )
            message( FATAL_ERROR 
                 "CUDA language enabled prior to setting CMAKE_CUDA_HOST_COMPILER. "
                 "Please set CMAKE_CUDA_HOST_COMPILER prior to "
                 "ENABLE_LANGUAGE(CUDA) or PROJECT(.. LANGUAGES CUDA)")
        endif()    
  
        if ( CMAKE_CXX_COMPILER )
            set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING "" FORCE)
        else()
            set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_C_COMPILER} CACHE STRING "" FORCE)
        endif()
    endif()
else()
   if (NOT CUDA_HOST_COMPILER)
        if("CUDA" IN_LIST _languages )
            message( FATAL_ERROR 
                 "CUDA language enabled prior to setting CUDA_HOST_COMPILER. "
                 "Please set CUDA_HOST_COMPILER prior to "
                 "ENABLE_LANGUAGE(CUDA) or PROJECT(.. LANGUAGES CUDA)")
        endif()    
        if ( CMAKE_CXX_COMPILER )
            set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING "" FORCE)
        else()
            set(CUDA_HOST_COMPILER ${CMAKE_C_COMPILER} CACHE STRING "" FORCE)
        endif()
    endif()
endif()


############################################################
# Basics
############################################################
enable_language(CUDA)


############################################################
# Map Legacy FindCUDA variables to native cmake variables
############################################################
# if we are linking with NVCC, define the link rule here
# Note that some mpi wrappers might have things like -Wl,-rpath defined, which when using 
# FindMPI can break nvcc. In that case, you should set ENABLE_FIND_MPI to Off and control
# the link using CMAKE_CUDA_LINK_FLAGS. -Wl,-rpath, equivalent would be -Xlinker -rpath -Xlinker
set(CMAKE_SHARED_LIBRARY_RPATH_LINK_CUDA_FLAG "-Xlinker -rpath -Xlinker")
set(CMAKE_CUDA_LINK_EXECUTABLE
  "${CMAKE_CUDA_COMPILER} <CMAKE_CUDA_LINK_FLAGS>  <FLAGS>  <LINK_FLAGS>  <OBJECTS> -o <TARGET>  <LINK_LIBRARIES>")
# do a no-op for the device links - for some reason the device link library dependencies are only a subset of the 
# executable link dependencies so the device link fails if there are any missing CUDA library dependencies. Since
# we are doing a link with the nvcc compiler, the device link step is unnecessary .
# Frustratingly, nvcc-link errors out if you pass it an empty file, so we have to first compile the empty file.
set(CMAKE_CUDA_DEVICE_LINK_LIBRARY "touch <TARGET>.cu ; ${CMAKE_CUDA_COMPILER} <CMAKE_CUDA_LINK_FLAGS> -std=c++11 --compile -rdc=false -Xcompiler -fPIC <TARGET>.cu -o <TARGET>")
set(CMAKE_CUDA_DEVICE_LINK_EXECUTABLE "touch <TARGET>.cu ; ${CMAKE_CUDA_COMPILER} <CMAKE_CUDA_LINK_FLAGS> -std=c++11 --compile -rdc=false -Xcompiler -fPIC <TARGET>.cu -o <TARGET>")

find_package(CUDA REQUIRED)

message(STATUS "CUDA Version:       ${CUDA_VERSION_STRING}")
message(STATUS "CUDA Compiler:      ${CMAKE_CUDA_COMPILER}")
if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.9.0" )
    message(STATUS "CUDA Host Compiler: ${CMAKE_CUDA_HOST_COMPILER}")
else()
    message(STATUS "CUDA Host Compiler: ${CUDA_HOST_COMPILER}")
endif()
message(STATUS "CUDA Include Path:  ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA Libraries:     ${CUDA_LIBRARIES}")
message(STATUS "CUDA Compile Flags: ${CMAKE_CUDA_FLAGS}")
message(STATUS "CUDA Link Flags:    ${CMAKE_CUDA_LINK_FLAGS}")
message(STATUS "CUDA Separable Compilation:  ${CUDA_SEPARABLE_COMPILATION}")
message(STATUS "CUDA Link with NVCC:         ${CUDA_LINK_WITH_NVCC}")

# don't propagate host flags - too easy to break stuff!
set (CUDA_PROPAGATE_HOST_FLAGS Off)
if (CMAKE_CXX_COMPILER)
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
else()
    set(CUDA_HOST_COMPILER ${CMAKE_C_COMPILER})
endif()

set(_cuda_compile_flags " ")
if (ENABLE_CLANG_CUDA)
    set (_cuda_compile_flags "-x cuda --cuda-gpu-arch=${BLT_CLANG_CUDA_ARCH} --cuda-path=${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "Clang CUDA Enabled. CUDA compile flags added: ${_cuda_compile_flags}")    
endif()

# depend on 'cuda', if you need to use cuda
# headers, link to cuda libs, and need to compile your
# source files with the cuda compiler (nvcc) instead of
# leaving it to the default source file language.
# This logic is handled in the blt_add_library/executable
# macros
# blt_register_library(NAME cuda
#                      COMPILE_FLAGS ${_cuda_compile_flags}
#                      INCLUDES ${CUDA_INCLUDE_DIRS}
#                      LIBRARIES ${CUDA_LIBRARIES})

# same as 'cuda' but we don't flag your source files as
# CUDA language.  This causes your source files to use 
# the regular C/CXX compiler. This is separate from 
# linking with nvcc.
# This logic is handled in the blt_add_library/executable
# macros
# blt_register_library(NAME cuda_runtime
#                      INCLUDES ${CUDA_INCLUDE_DIRS}
#                      LIBRARIES ${CUDA_LIBRARIES})
