cmake_minimum_required (VERSION 3.8.2)

project (tao_ops)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CMake/TaoHelpers.cmake)

# workaround cmake empty CMAKE_CUDA_DEVICE_LINK_LIBRARY issues
# solution derived from the following:
# https://ecpci-test.alcf.anl.gov/jgalaro/calc_pi/-/blob/7d7d0582d339ee5ccaf656f903b73977abce8b9e/blt/master_20Feb19/cmake/thirdparty/SetupCUDA.cmake
if(${TAO_CUDA})
  include(CMake/SetupCUDA.cmake)
endif()

if(${TAO_ROCM})
  include(CMake/SetupROCM.cmake)
endif()

if(EXISTS ${PROJECT_BINARY_DIR}/config.cmake)
    include(${PROJECT_BINARY_DIR}/config.cmake)
endif()

############## options ################
tao_option(TAO_ENABLE_WARMUP_XFLOW "Build xflow for warming up or not." OFF)
tao_option(TAO_DISABLE_LOWER_IF_WHILE "If disable lowering if & while." OFF)
tao_option(TAO_ENABLE_CXX_TESTING "If ON, TAO tests will be built and run." OFF)
tao_option(TAO_ENABLE_UPLOAD_TOOL "if ON, enable uploading tool building" OFF)
tao_option(TAO_DISABLE_LINK_TF_FRAMEWORK "if disable, do not link tf framework" OFF)
tao_option(TAO_ENABLE_AUTH "if ON, linking the authentication library" OFF)
tao_option(PLATFORM_ALIBABA "if ON, building with internal library" OFF)
tao_option(ENABLE_BLADE_GEMM "if ON, using blade_gemm for gpu dot-general" OFF)
tao_option(TAO_ENABLE_MKLDNN "if ON, using mkldnn for compute-intensive ops" OFF)
set(MKL_ROOT "third_party/mkldnn/build/intel" CACHE STRING "Set to mkl_root if enable MKL")
tao_option(GTEST_TAR_URL "download cmake tarball via URL" "https://github.com/google/googletest/archive/refs/tags/release-1.11.0.tar.gz")
tao_option(ROCM_PATH "Path to the ROCM toolchain" /opt/rocm)

############## config start ##############
if (NOT UNIX)
    message(FATAL_ERROR "TAO compiles on linux only for now, but your are on " ${CMAKE_SYSTEM_NAME})
endif()

####### include helper functions #########
list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_LIST_DIR}/CMake
)
include(TaoHelpers)

############## find python ##############
if(NOT DEFINED PYTHON)
    if (${CMAKE_VERSION} VERSION_LESS 3.12.0)
        find_package(PythonInterp)
        find_package(PythonLibs)
        set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
        set(Python_LIBRARIES ${PYTHON_LIBRARIES})
        set(Python_FOUND (${PYTHONLIBS_FOUND} AND ${PYTHONINTERP_FOUND}))
    else()
        find_package(Python COMPONENTS Development Interpreter REQUIRED)
    endif()

    if(NOT DEFINED Python_EXECUTABLE)
        set(PYTHON "python")
    else()
        set(PYTHON ${Python_EXECUTABLE})
    endif()
else()
    execute_process(
      COMMAND
        "ldd" "${PYTHON}"
      COMMAND
        "grep" "python"
      COMMAND
        "awk" "{print $3}"
      OUTPUT_VARIABLE
        Python_LIBRARIES
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()
message(STATUS "Python command: " ${PYTHON})
message(STATUS "Python library: " ${Python_LIBRARIES})

########### find tensorflow ############
find_package(Tensorflow REQUIRED)

############# find protoc ##############
find_package(Protobuf ${Tensorflow_PB_VERSION}
    PATHS ${CMAKE_CURRENT_BINARY_DIR}/.download_cache
    QUIET
)
if(NOT DEFINED Protobuf_PROTOC_EXECUTABLE)
    if(${TAO_AARCH64})
        tao_execute_cmd('
            DESC "download protoc"
            PWD "."
            COMMAND bash ${tao_ops_SOURCE_DIR}/tools/download_protoc.sh ${CMAKE_CURRENT_BINARY_DIR} ${Tensorflow_PB_VERSION} "linux-aarch_64"
            STD_OUT Protobuf_PROTOC_EXECUTABLE
        )
    else()
        tao_execute_cmd('
            DESC "download protoc"
            PWD "."
            COMMAND bash ${tao_ops_SOURCE_DIR}/tools/download_protoc.sh ${CMAKE_CURRENT_BINARY_DIR} ${Tensorflow_PB_VERSION} "linux-x86_64"
            STD_OUT Protobuf_PROTOC_EXECUTABLE
        )
    endif()
endif()
message(STATUS "Download protoc to:\n   " ${Protobuf_PROTOC_EXECUTABLE})


############ tao ops flags ############

# -DTF_X_XX
tao_get_tf_version_def(
    VERSION ${Tensorflow_VERSION}
    VERSION_DEF TAO_OPS_DEFINITIONS
)

# -DIS_PAI_TF
if(${Tensorflow_IS_PAI})
    message("Define preprocess macro: -DIS_PAI_TF")
    list(APPEND TAO_OPS_DEFINITIONS "-DIS_PAI_TF")
endif()

# -DDISABLE_LOWER_IF_WHILE
if(${TAO_DISABLE_LOWER_IF_WHILE})
    list(APPEND TAO_OPS_DEFINITIONS "-DDISABLE_LOWER_IF_WHILE")
    message(STATUS "Disable lowering if & while.")
endif()

if(${TAO_CUDA})
    list(APPEND TAO_OPS_DEFINITIONS "-DGOOGLE_CUDA")
    message(STATUS "Define preprocess macro: -DGOOGLE_CUDA")
endif()

if(${TAO_DCU})
    list(APPEND TAO_OPS_DEFINITIONS "-DTENSORFLOW_USE_DCU")
    message(STATUS "Define preprocess macro: -DTENSORFLOW_USE_DCU")
    list(APPEND TAO_OPS_DEFINITIONS "-DTENSORFLOW_USE_ROCM")
    message(STATUS "Define preprocess macro: -DTENSORFLOW_USE_ROCM")
endif()

if(${TAO_ROCM})
  list(APPEND TAO_OPS_DEFINITIONS "-DTENSORFLOW_USE_ROCM")
  message(STATUS "Define preprocess macro: -DTENSORFLOW_USE_ROCM")
endif()

message(STATUS "TAO_OPS_DEFINITIONS:")
foreach(tao_ops_def ${TAO_OPS_DEFINITIONS})
    message(STATUS "  ${tao_ops_def}")
endforeach(tao_ops_def)
add_definitions(${TAO_OPS_DEFINITIONS})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# TODO(xiafei.qiuxf): add debug and release profile
string(REPLACE " " ";" TF_CFLAGS ${Tensorflow_CFLAGS})
add_compile_options(-g -DNDEBUG -O3 ${TF_CFLAGS})
# Report all warning as error for now except `sign-compare` and `unused-variable`
# since such code exits in headers of lower version (1.12) tf.
#add_compile_options(-Wall -Werror -Wno-error=sign-compare -Wno-error=unused-variable)

if(${TAO_CPU_ONLY})
  add_compile_options(-DTAO_CPU_ONLY)
endif()

if(${TAO_X86})
  add_compile_options(-DTAO_X86)
endif()

if(${TAO_AARCH64})
  add_compile_options(-DTAO_AARCH64)
endif()

if(${BLAZE_OPT})
  add_compile_options(-DBLAZE_OPT)
endif()

if(${TAO_ENABLE_UPLOAD_TOOL} AND ${PLATFORM_ALIBABA})
  add_compile_options(-DTAO_ENABLE_UPLOAD_TOOL)
endif()

if(${ENABLE_BLADE_GEMM})
  add_compile_options(-DENABLE_BLADE_GEMM)
endif()

if(${TAO_ENABLE_MKLDNN})
  add_compile_options(-DTAO_ENABLE_MKLDNN)
endif()

if (${PLATFORM_ALIBABA})
  message("Build DISC in Alibaba platform")
  add_compile_options(-DPLATFORM_ALIBABA)
  get_filename_component(PLATFORM_ALIBABA_DIR ../../platform_alibaba ABSOLUTE)
endif()

include_directories(${tao_ops_SOURCE_DIR})

############### test ###############
if(${TAO_ENABLE_CXX_TESTING})
  # enable CTest.  This will set BUILD_TESTING to ON unless otherwise specified
  # on the command line
  include(CTest)
  enable_testing()
endif()

## check targets
if(BUILD_TESTING)
    include(CMake/Googletest/DownloadGTest.cmake)
    set(absl_gtest_src_dir ${CMAKE_BINARY_DIR}/googletest-src)
    set(absl_gtest_build_dir ${CMAKE_BINARY_DIR}/googletest-build)

    tao_check_target(gtest)
    tao_check_target(gtest_main)
    tao_check_target(gmock)

    list(APPEND TAO_TEST_COMMON_LIBRARIES
        gtest_main
        gtest
        gmock
        ${CMAKE_THREAD_LIBS_INIT}
    )
endif()

############### test end ###############

############### libtao_ops.so ###############

set(OBJS
    $<TARGET_OBJECTS:tao_bridge>
    $<TARGET_OBJECTS:tf_port>
    $<TARGET_OBJECTS:passes>
    $<TARGET_OBJECTS:ops>
    $<TARGET_OBJECTS:kernels>
    $<TARGET_OBJECTS:ral>
    $<TARGET_OBJECTS:mlir>
)

if(NOT ${TAO_CPU_ONLY} AND ${PLATFORM_ALIBABA})
  list(APPEND OBJS $<TARGET_OBJECTS:gpu>)
endif()

if(${PLATFORM_ALIBABA})
  list(APPEND OBJS $<TARGET_OBJECTS:tao_launch_op>)
  list(APPEND OBJS $<TARGET_OBJECTS:transform>)
endif()

add_library(tao_ops SHARED ${OBJS})
if(NOT ${TAO_DISABLE_LINK_TF_FRAMEWORK})
  target_link_libraries(tao_ops ${Tensorflow_LIBRARIES})
endif()


if(${TAO_CUDA} AND ${PLATFORM_ALIBABA})
  target_link_libraries(tao_ops bace)
endif()

target_link_options(tao_ops PUBLIC "-Wl,--exclude-libs,ALL,-ldl")

# required by uploader in tao_bridge
if(${TAO_ENABLE_UPLOAD_TOOL} AND ${PLATFORM_ALIBABA})
    include_directories(${PLATFORM_ALIBABA_DIR}/third_party/aliyun-oss-cpp-sdk/sdk/include)
    add_subdirectory(${PLATFORM_ALIBABA_DIR}/third_party/aliyun-oss-cpp-sdk aliyun-oss-cpp-sdk)
    add_subdirectory(${PLATFORM_ALIBABA_DIR}/tools/tao/uploader uploader)
endif()

add_subdirectory(tao_bridge)

if(${ENABLE_BLADE_GEMM})
  add_subdirectory(${PLATFORM_ALIBABA_DIR}/blade_gemm blade_gemm EXCLUDE_FROM_ALL)
endif()
if(${TAO_ENABLE_MKLDNN})
  add_subdirectory(third_party/mkldnn dnnl)
endif()
# Unset these CACHE variables to not change the blade project structure
unset(CMAKE_ARCHIVE_OUTPUT_DIRECTORY CACHE)
unset(CMAKE_LIBRARY_OUTPUT_DIRECTORY CACHE)
unset(CMAKE_PDB_OUTPUT_DIRECTORY CACHE)
unset(CMAKE_RUNTIME_OUTPUT_DIRECTORY CACHE)
if(${TAO_CPU_ONLY})
  target_link_libraries(tao_ops gomp)
endif()
if(${ENABLE_BLADE_GEMM})
  target_link_libraries(tao_ops blade_gemm)
endif()
if(${TAO_ENABLE_MKLDNN})
  target_link_libraries(tao_ops dnnl)
  if(${TAO_X86})
    set(MKL_LINK_OPTION "-Wl,--start-group ${MKL_ROOT}/lib/libmkl_intel_ilp64.a ${MKL_ROOT}/lib/libmkl_gnu_thread.a ${MKL_ROOT}/lib/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl")
    target_link_libraries(tao_ops ${MKL_LINK_OPTION})
  endif()
endif()
