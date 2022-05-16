################################
# Sanity Checks
################################

# Rare case of two flags being incompatible
# if (DEFINED CMAKE_SKIP_BUILD_RPATH AND DEFINED HIP_LINK_WITH_NVCC)
#   if (NOT CMAKE_SKIP_BUILD_RPATH AND HIP_LINK_WITH_NVCC)
#     message( FATAL_ERROR
#       "CMAKE_SKIP_BUILD_RPATH (FALSE) and HIP_LINK_WITH_NVCC (TRUE) "
#       "are incompatible when linking explicit shared libraries. Set "
#       "CMAKE_SKIP_BUILD_RPATH to TRUE.")
#   endif()
# endif()

# HIP_HOST_COMPILER was changed in 3.9.0 to CMAKE_HIP_HOST_COMPILER and
# needs to be set prior to enabling the HIP language
get_property(_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.9.0" )
    if ( NOT CMAKE_HIP_HOST_COMPILER )
        if("HIP" IN_LIST _languages )
            message( FATAL_ERROR 
                 "HIP language enabled prior to setting CMAKE_HIP_HOST_COMPILER. "
                 "Please set CMAKE_HIP_HOST_COMPILER prior to "
                 "ENABLE_LANGUAGE(HIP) or PROJECT(.. LANGUAGES HIP)")
        endif()    
  
        if ( CMAKE_CXX_COMPILER )
            set(CMAKE_HIP_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING "" FORCE)
        else()
            set(CMAKE_HIP_HOST_COMPILER ${CMAKE_C_COMPILER} CACHE STRING "" FORCE)
        endif()
    endif()
else()
   if (NOT HIP_HOST_COMPILER)
        if("HIP" IN_LIST _languages )
            message( FATAL_ERROR 
                 "HIP language enabled prior to setting HIP_HOST_COMPILER. "
                 "Please set HIP_HOST_COMPILER prior to "
                 "ENABLE_LANGUAGE(HIP) or PROJECT(.. LANGUAGES HIP)")
        endif()    
        if ( CMAKE_CXX_COMPILER )
            set(HIP_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING "" FORCE)
        else()
            set(HIP_HOST_COMPILER ${CMAKE_C_COMPILER} CACHE STRING "" FORCE)
        endif()
    endif()
endif()


############################################################
# Basics
############################################################
MESSAGE("Before find hip.")

enable_language(HIP)

############################################################
MESSAGE("After find hip.")

############################################################
# Map Legacy FindHIP variables to native cmake variables
############################################################
# if we are linking with NVCC, define the link rule here
# Note that some mpi wrappers might have things like -Wl,-rpath defined, which when using 
# FindMPI can break nvcc. In that case, you should set ENABLE_FIND_MPI to Off and control
# the link using CMAKE_HIP_LINK_FLAGS. -Wl,-rpath, equivalent would be -Xlinker -rpath -Xlinker
set(CMAKE_SHARED_LIBRARY_RPATH_LINK_HIP_FLAG "-Xlinker -rpath -Xlinker")
set(CMAKE_HIP_LINK_EXECUTABLE
  "${CMAKE_HIP_COMPILER} <CMAKE_HIP_LINK_FLAGS>  <FLAGS>  <LINK_FLAGS>  <OBJECTS> -o <TARGET>  <LINK_LIBRARIES>")
# do a no-op for the device links - for some reason the device link library dependencies are only a subset of the 
# executable link dependencies so the device link fails if there are any missing HIP library dependencies. Since
# we are doing a link with the nvcc compiler, the device link step is unnecessary .
# Frustratingly, nvcc-link errors out if you pass it an empty file, so we have to first compile the empty file.
set(CMAKE_HIP_DEVICE_LINK_LIBRARY "touch <TARGET>.cu ; ${CMAKE_HIP_COMPILER} <CMAKE_HIP_LINK_FLAGS> -std=c++11 --compile -rdc=false -Xcompiler -fPIC <TARGET>.cu -o <TARGET>")
set(CMAKE_HIP_DEVICE_LINK_EXECUTABLE "touch <TARGET>.cu ; ${CMAKE_HIP_COMPILER} <CMAKE_HIP_LINK_FLAGS> -std=c++11 --compile -rdc=false -Xcompiler -fPIC <TARGET>.cu -o <TARGET>")

find_package(HIP REQUIRED)

message(STATUS "HIP Version:       ${HIP_VERSION_STRING}")
message(STATUS "HIP Compiler:      ${CMAKE_HIP_COMPILER}")
if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.9.0" )
    message(STATUS "HIP Host Compiler: ${CMAKE_HIP_HOST_COMPILER}")
else()
    message(STATUS "HIP Host Compiler: ${HIP_HOST_COMPILER}")
endif()
message(STATUS "HIP Include Path:  ${HIP_INCLUDE_DIRS}")
message(STATUS "HIP Libraries:     ${HIP_LIBRARIES}")
message(STATUS "HIP Compile Flags: ${CMAKE_HIP_FLAGS}")
message(STATUS "HIP Link Flags:    ${CMAKE_HIP_LINK_FLAGS}")
message(STATUS "HIP Separable Compilation:  ${HIP_SEPARABLE_COMPILATION}")
message(STATUS "HIP Link with NVCC:         ${HIP_LINK_WITH_NVCC}")

# don't propagate host flags - too easy to break stuff!
set (HIP_PROPAGATE_HOST_FLAGS Off)
if (CMAKE_CXX_COMPILER)
    set(HIP_HOST_COMPILER ${CMAKE_CXX_COMPILER})
else()
    set(HIP_HOST_COMPILER ${CMAKE_C_COMPILER})
endif()

set(_HIP_compile_flags " ")
if (ENABLE_CLANG_HIP)
    set (_HIP_compile_flags "-x HIP --HIP-gpu-arch=${BLT_CLANG_HIP_ARCH} --HIP-path=${HIP_TOOLKIT_ROOT_DIR}")
    message(STATUS "Clang HIP Enabled. HIP compile flags added: ${_HIP_compile_flags}")    
endif()

# depend on 'HIP', if you need to use HIP
# headers, link to HIP libs, and need to compile your
# source files with the HIP compiler (nvcc) instead of
# leaving it to the default source file language.
# This logic is handled in the blt_add_library/executable
# macros
# blt_register_library(NAME HIP
#                      COMPILE_FLAGS ${_HIP_compile_flags}
#                      INCLUDES ${HIP_INCLUDE_DIRS}
#                      LIBRARIES ${HIP_LIBRARIES})

# same as 'HIP' but we don't flag your source files as
# HIP language.  This causes your source files to use 
# the regular C/CXX compiler. This is separate from 
# linking with nvcc.
# This logic is handled in the blt_add_library/executable
# macros
# blt_register_library(NAME HIP_runtime
#                      INCLUDES ${HIP_INCLUDE_DIRS}
#                      LIBRARIES ${HIP_LIBRARIES})
