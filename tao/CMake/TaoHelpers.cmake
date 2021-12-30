include(CMakeParseArguments)

macro(tao_option variable description value)
  if(NOT DEFINED ${variable})
    set(${variable} ${value} CACHE STRING ${description})
  endif()
endmacro()

# Simple wrapper to execute_process
function(tao_execute_cmd)
  cmake_parse_arguments(TAO
    ""
    "DESC;STD_OUT;STD_ERR;PWD"
    "COMMAND"
    ${ARGN})
  execute_process(
    COMMAND ${TAO_COMMAND}
    WORKING_DIRECTORY ${TAO_PWD}
    RESULT_VARIABLE CMD_RESULT
    OUTPUT_VARIABLE OUT
    ERROR_VARIABLE ERR
  )
  if (NOT CMD_RESULT EQUAL "0")
    message(FATAL_ERROR "`${TAO_DESC}` failed: ${CMD_RESULT}\n"
                        "cmd   : ${TAO_COMMAND} \n"
                        "stdout: ${OUT}" "\n"
                        "stderr: ${ERR}")
  endif()

  set(${TAO_STD_OUT} ${OUT} PARENT_SCOPE)
  set(${TAO_STD_ERR} ${ERR} PARENT_SCOPE)
endfunction()

# add a tao cxx test target.
function(tao_cc_test)
  if(NOT BUILD_TESTING)
    return()
  endif()

  cmake_parse_arguments(TAO_CC_TEST
    ""
    "NAME"
    "SRCS;COPTS;DEFINES;EXTRALIBS"
    ${ARGN}
  )

  set(_NAME "tao_${TAO_CC_TEST_NAME}")
  add_executable(${_NAME} "")
  target_sources(${_NAME} PRIVATE ${TAO_CC_TEST_SRCS})
  target_include_directories(${_NAME}
    # PUBLIC ${ABSL_COMMON_INCLUDE_DIRS}
    PRIVATE ${GMOCK_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS}
  )
  target_compile_definitions(${_NAME}
    PUBLIC ${TAO_CC_TEST_DEFINES} 
  )
  target_compile_options(${_NAME}
    PRIVATE ${TAO_CC_TEST_COPTS}
  )
  # for simplicity we just link to the whole tao_ops shared libirary.
  target_link_libraries(${_NAME}
     gmock gtest gtest_main tao_ops ${TAO_CC_TEST_EXTRALIBS}
  )
  # Add all Abseil targets to a a folder in the IDE for organization.
#   set_property(TARGET ${_NAME} PROPERTY FOLDER ${ABSL_IDE_FOLDER}/test)
#   set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD ${ABSL_CXX_STANDARD})
#   set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

  add_test(NAME ${_NAME} COMMAND ${_NAME})
endfunction()


# Get tf versioin and flags string.
function(tao_get_tf_flags)
  cmake_parse_arguments(TAO
    ""
    "VERSION;IS_PAI;CFLAGS;LDFLAGS"
    ""
    ${ARGN})
  tao_execute_cmd(
      DESC "get tensorflow version"
      PWD "."
      COMMAND ${PYTHON} -c "from __future__ import print_function\nimport tensorflow as tf\nprint(tf.__version__, end = '')"
      STD_OUT OUTPUT
  )
  set(${TAO_VERSION} ${OUTPUT} PARENT_SCOPE)
  if("${OUTPUT}" MATCHES "PAI")
    set(${TAO_IS_PAI} ON PARENT_SCOPE)
  else()
    set(${TAO_IS_PAI} OFF PARENT_SCOPE)
  endif()
  tao_execute_cmd(
      DESC "get tensorflow compile flags"
      PWD "."
      COMMAND ${PYTHON} -c "from __future__ import print_function\nimport tensorflow as tf\nprint(' '.join(tf.sysconfig.get_compile_flags()), end = '')"
      STD_OUT OUTPUT
  )
  set(${TAO_CFLAGS} ${OUTPUT} PARENT_SCOPE)
  tao_execute_cmd(
    DESC "get tensorflow link flags"
    PWD "."
    COMMAND ${PYTHON} -c "from __future__ import print_function\nimport tensorflow as tf\nprint(' '.join(tf.sysconfig.get_link_flags()), end = '')"
    STD_OUT OUTPUT
  )
  set(${TAO_LDFLAGS} ${OUTPUT} PARENT_SCOPE)
endfunction()

# Get version definition from version string: 
#   1.15.2 -> TF_1_15
#   1.12.2PAI -> TF_1_12
function(tao_get_tf_version_def)
  cmake_parse_arguments(TAO
  ""
  "VERSION;VERSION_DEF"
  ""
  ${ARGN})
  string(REGEX MATCH "^[0-9]+\.[0-9]+" MAJAR_MINOR ${TAO_VERSION})
  string(REPLACE "." "_" PARTIAL_DEF ${MAJAR_MINOR})
  set(TF_DEF "-DTF_")
  string(APPEND TF_DEF ${PARTIAL_DEF})
  set(${TAO_VERSION_DEF} ${TF_DEF} PARENT_SCOPE)
endfunction()

# Check for a target, exit with error if not exist.
function(tao_check_target my_target)
  if(NOT TARGET ${my_target})
    message(FATAL_ERROR "Required CMake target ${my_target} not found.")
  endif(NOT TARGET ${my_target})
endfunction()

# Check if dependent pb is in google::protobuf3 namespace from cflags.
function(tao_is_pb3)
  if(DEFINED TAO_IS_PB3)
    return()
  endif()
  cmake_parse_arguments(TAO
    ""
    "CFLAG;OUTPUT"
    ""
    ${ARGN})

  # message("CLFAG: ${TAO_CFLAG}")
  string(REGEX MATCH "\-I[^ ]+" SHORT_FLAG "${TAO_CFLAG}")
  # message("SHORT_FLAG: ${SHORT_FLAG}")
  string(SUBSTRING ${SHORT_FLAG} 2 -1 CHECK_PATH)
  if("${CHECK_PATH}" STREQUAL "")
    message(FATAL_ERROR "Failed to extract include path from cflags: ${TAO_CFLAG} .
Please specify TAO_IS_PB3 to ON or OFF explicitly.")
  endif()

  string(APPEND CHECK_PATH "/google/protobuf/stubs/common.h")
  message(STATUS "Infer pb version from file: ${CHECK_PATH}")

  file(READ ${CHECK_PATH} CONTENT)
  if("${CONTENT}" MATCHES "namespace protobuf3 {")
    set(${TAO_OUTPUT} ON PARENT_SCOPE)
  else()
    set(${TAO_OUTPUT} OFF PARENT_SCOPE)
  endif()
endfunction()

function(tao_prepare_xflow_sdk)
  cmake_parse_arguments(TAO
    ""
    "HEADER_DIR;"
    ""
    ${ARGN})
  set(XFLOW_SRC_DIR "${PROJECT_SOURCE_DIR}/../../platform_alibaba/third_party/algorithms")
  if(NOT EXISTS ${XFLOW_SRC_DIR})
    message(FATAL_ERROR "xflow directory (${XFLOW_SRC_DIR}) does not exits, set TAO_ENABLE_WARMUP_XFLOW to OFF or \
                        run `git submodule update --init --recursive` to fetch xflow source code.")
  endif()
  tao_execute_cmd(
    DESC "Build algorithms.git"
    PWD "${XFLOW_SRC_DIR}"
    COMMAND "bin/scons-local/scons.py" "-j8"
    STD_OUT OUTPUT
  )
  set(${TAO_HEADER_DIR} "${XFLOW_SRC_DIR}/sdk/include" PARENT_SCOPE)
  message(STATUS "Prparing xflow SDK done, header dir: ${TAO_HEADER_DIR}")
endfunction()

function(tao_validate_cxx_version tf_flags)
  if(${CMAKE_COMPILER_IS_GNUCC} 
      AND ${tf_flags} MATCHES "-D_GLIBCXX_USE_CXX11_ABI=1"
      AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 5.1)
    message(FATAL_ERROR "Target tensorflow requires CXX11 ABI but your "
      "GCC version is too low: ${CMAKE_CXX_COMPILER_VERSION}")
  endif()
endfunction()

# turn tao build env vars as cmake var.
function(tao_export_build_env_var)
  list(APPEND BUILD_VARS TAO_BUILD_VERSION
                         TAO_BUILD_GIT_BRANCH
                         TAO_BUILD_GIT_HEAD
                         TAO_BUILD_HOST
                         TAO_BUILD_IP
                         TAO_BUILD_TIME)
  foreach(v ${BUILD_VARS})
    if("$ENV{${v}}" STREQUAL "")
      set(${v} "UNKNOWN" PARENT_SCOPE)
    else()
      set(${v} "$ENV{${v}}" PARENT_SCOPE)
    endif()
  endforeach()

endfunction()
