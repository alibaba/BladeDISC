# - Try to find Tensorflow
# Once done, this will define
#
# Tensorflow_FOUND - system has Tensorflow
# Tensorflow_INCLUDE_DIRS - the Tensorflow include directories
# Tensorflow_LIBRARIES - the libtensorflow_framework.so library target
# Tensorflow_EXTRA_LIBRARIES - the _pywrap_tensorflow_internal.so library target
# Tensorflow_LIBRARY_DIRS - tensorflow library path
# Tensorflow_EXTRA_LIBRARY_DIRS - extra tensorflow library path
# Tensorflow_DEFINITIONS - tensorflow definitions
# Tensorflow_IS_PAI - whether tensorflow is prebuilt with pai
# Tensorflow_USE_PB3 - whether tensorflow is prebuilt with protobuf3
# Tensorflow_PB_VERSION - protobuf version used in prebuilt tensorflow


if (NOT ${Python_FOUND})
  message(FATAL_ERROR "Python should have been found before tensorflow")
endif()

# Somehow Python_SITELIB may not work properly, find by ourselves
if (NOT ${Python_SITELIB})
  set(SITELIB_CMD "from distutils import sysconfig\nprint(sysconfig.get_python_lib(plat_specific=False, standard_lib=False), end='')")
  execute_process(
    COMMAND
      "${Python_EXECUTABLE}" "-c" "${SITELIB_CMD}"
    OUTPUT_VARIABLE
      Python_SITELIB
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endif()

tao_get_tf_flags(
    VERSION Tensorflow_VERSION
    IS_PAI Tensorflow_IS_PAI
    CFLAGS Tensorflow_CFLAGS
    LDFLAGS Tensorflow_LDFLAGS
)

tao_validate_cxx_version("${Tensorflow_CFLAGS}")

string(REGEX MATCH "-L([^ ]*)" RAW_TF_LINK_DIRS "${Tensorflow_LDFLAGS}")
string(REGEX MATCH "-l([^ ]*)" RAW_TF_LINK_LIBS "${Tensorflow_LDFLAGS}")
string(REGEX MATCH "-I([^ ]*)" RAW_TF_INCLUDE_DIRS "${Tensorflow_CFLAGS}")
string(REGEX MATCH "-D([^ ]*)" RAW_TF_DEFINITIONS "${Tensorflow_CFLAGS}")
string(SUBSTRING ${RAW_TF_LINK_DIRS} 2 -1 Tensorflow_LIBRARY_DIRS)
string(SUBSTRING ${RAW_TF_INCLUDE_DIRS} 2 -1 Tensorflow_INCLUDE_DIRS)
string(SUBSTRING ${RAW_TF_DEFINITIONS} 2 -1 Tensorflow_DEFINITIONS)
string(REGEX REPLACE "-l[:]?([^ ]*)" "\\1" Tensorflow_LIBNAME ${RAW_TF_LINK_LIBS})

# Check protobuf version in prebuilt tensorflow
string(CONCAT CHECK_PATH
    ${Tensorflow_INCLUDE_DIRS}
    "/google/protobuf/stubs/common.h"
)

file(READ ${CHECK_PATH} CONTENT)
string(REGEX MATCH
    "#define GOOGLE_PROTOBUF_VERSION [0-9]+"
    PB_VER
    ${CONTENT}
)
string(REGEX REPLACE "[^0-9]+([0-9]+)$" "\\1" PB_VER_NUM ${PB_VER})
math(EXPR PB_MAJOR_VER "${PB_VER_NUM}/1000000")
math(EXPR PB_MINOR_VER "${PB_VER_NUM}/1000-${PB_MAJOR_VER}*1000")
math(EXPR PB_MICRO_VER "${PB_VER_NUM}-${PB_MAJOR_VER}*1000000-${PB_MINOR_VER}*1000")
set(Tensorflow_PB_VERSION "${PB_MAJOR_VER}.${PB_MINOR_VER}.${PB_MICRO_VER}")

if("${CONTENT}" MATCHES "namespace protobuf3 {")
    set(Tensorflow_USE_PB3 ON)
else()
    set(Tensorflow_USE_PB3 OFF)
endif()

find_library(Tensorflow_LIBRARIES
    ${Tensorflow_LIBNAME}
    ${Tensorflow_LIBRARY_DIRS}
)

set(Tensorflow_EXTRA_LIBRARY_DIRS ${Tensorflow_LIBRARY_DIRS}/python)
find_library(Tensorflow_EXTRA_LIBRARIES
    "_pywrap_tensorflow_internal.so"
    ${Tensorflow_EXTRA_LIBRARY_DIRS}
)

message(STATUS "Tensorflow version       : " ${Tensorflow_VERSION})
message(STATUS "Tensorflow include dir   : " ${Tensorflow_INCLUDE_DIRS})
message(STATUS "Tensorflow lib           : " ${Tensorflow_LIBRARIES})
message(STATUS "Tensorflow lib dir       : " ${Tensorflow_LIBRARY_DIRS})
message(STATUS "Tensorflow extra lib     : " ${Tensorflow_EXTRA_LIBRARIES})
message(STATUS "Tensorflow extra lib dir : " ${Tensorflow_EXTRA_LIBRARY_DIRS})
message(STATUS "Tensorflow definitions   : " ${Tensorflow_DEFINITIONS})
message(STATUS "Tensorflow is pai        : " ${Tensorflow_IS_PAI})
message(STATUS "Tensorflow use pb3       : " ${Tensorflow_USE_PB3})
message(STATUS "Tensorflow pb version    : " ${Tensorflow_PB_VERSION})

