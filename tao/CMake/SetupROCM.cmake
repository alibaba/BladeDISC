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

enable_language(HIP)


set(CMAKE_SHARED_LIBRARY_RPATH_LINK_HIP_FLAG "-Xlinker -rpath -Xlinker")
set(CMAKE_HIP_LINK_EXECUTABLE
  "${CMAKE_HIP_COMPILER} <CMAKE_HIP_LINK_FLAGS>  <FLAGS>  <LINK_FLAGS>  <OBJECTS> -o <TARGET>  <LINK_LIBRARIES>")

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
