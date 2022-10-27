project("blade_gemm" CXX)
cmake_minimum_required (VERSION 3.8.2)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 14)
set(USE_TVM CACHE BOOL OFF)
set(ROCM_PATH "/opt/rocm" CACHE STRING "rocm ToolKit path" FORCE)
set(BLADE_GEMM_NVCC_ARCHS 75;80 CACHE STRING "Wrapper for CUTLASS_NVCC_ARCHS")
set(BLADE_GEMM_LIBRARY_KERNELS s1688tf32gemm,f16_s1688gemm_f16,f16_s16816gemm_f16,s16816tf32gemm,s8_i8816gemm_s8,s1688tf32fprop_,f16_s1688fprop_*_f16,,s8_i16832fprop_optimized_s8,s8_i8816fprop_optimized_s8,f16_s16816fprop_optimized_f16,f16_s884fprop_optimized_f16,f16_s16816fprop_optimized_f16,s16832spgemm,f16_s884fprop_*optimized_f16,f16_s16816fprop_*optimized_f16 CACHE STRING "Wrapper for CUTLASS_LIBRARY_KERNELS")

set(CUTLASS_NVCC_ARCHS ${BLADE_GEMM_NVCC_ARCHS} CACHE STRING "set for CUTLASS_NVCC_ARCHS" FORCE)
set(CUTLASS_LIBRARY_KERNELS ${BLADE_GEMM_LIBRARY_KERNELS} CACHE STRING "set for CUTLASS_LIBRARY_KERNELS" FORCE)
set(CUTLASS_ENABLE_TESTS CACHE BOOL OFF FORCE)
set(CUTLASS_UNITY_BUILD_ENABLED CACHE BOOL ON FORCE)

message(STATUS "Build with cutlass flags: CUTLASS_LIBRARY_KERNELS: ${CUTLASS_NVCC_ARCHS}")
message(STATUS "Build with cutlass flags: CUTLASS_LIBRARY_KERNELS: ${CUTLASS_LIBRARY_KERNELS}")
message(STATUS "Build with cutlass flags: CUTLASS_ENABLE_TESTS: ${CUTLASS_ENABLE_TESTS}")
message(STATUS "Build with cutlass flags: CUTLASS_UNITY_BUILD_ENABLED: ${CUTLASS_UNITY_BUILD_ENABLED}")
message(STATUS "Build with TVM ${USE_TVM}")
message(STATUS "Build with ROCM LIB PATH ${ROCM_PATH}")


if(USE_TVM) 
  include(ExternalProject)
  include(${CMAKE_SOURCE_DIR}/third_party/json.cmake)
endif()

if(NOT USE_TVM)
  find_package(CUDA REQUIRED)
  set(CMAKE_CUDA_ARCHITECTURES 75;80)
  add_subdirectory(cutlass_library cutlass EXCLUDE_FROM_ALL)
endif()

file(GLOB LIB_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cc)

if(USE_TVM) 
  set(CMAKE_CXX_COMPILER "/usr/bin/hipcc" CACHE STRING "rocm hipcc path" FORCE)
  file(GLOB_RECURSE LIB_ROCM_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/tvm/*.cc)
  list(APPEND LIB_SOURCES
    ${LIB_ROCM_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rotlass/rotlass_gen/rotlass_handle.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rotlass/rotlass_gen/handle.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rotlass/rotlass_gen/gen_ops.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rotlass/rotlass_gen/gen_double_0_0.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rotlass/rotlass_gen/gen_double_0_1.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rotlass/rotlass_gen/gen_double_1_0.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rotlass/rotlass_gen/gen_float_0_0.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rotlass/rotlass_gen/gen_float_0_1.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rotlass/rotlass_gen/gen_float_1_0.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rotlass/rotlass_gen/gen_double_0_0_64.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rotlass/rotlass_gen/gen_double_0_1_64.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rotlass/rotlass_gen/gen_double_1_0_64.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rotlass/rotlass_gen/gen_float_0_0_32.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rotlass/rotlass_gen/gen_float_0_1_32.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rotlass/rotlass_gen/gen_float_1_0_32.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rotlass/rotlass_gen/util.cc
  )
  MESSAGE("LIB_SOURCES ${LIB_SOURCES}")
  include_directories(
    include
    third_party/rotlass/include
    third_party/rotlass/tools/util/include
    third_party/rotlass/tools/library/include
    third_party/rotlass/tools/library/src
    third_party/rotlass/third_party/rocWMMA/library/include
    third_party/rotlass/rotlass_gen
    ${CMAKE_BINARY_DIR}/third_party/json/single_include
    ${ROCM_PATH}/include)
  add_compile_options(-D__HIP_PLATFORM_AMD__ -DCUTLASS_ARCH_WMMA_ENABLED -DCUTLASS_ARCH_WMMA_SM70_ENABLED)
else()
  file(GLOB_RECURSE LIB_CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/cutlass/*.cc)
  list(APPEND LIB_SOURCES ${LIB_CUDA_SOURCES})
  list(APPEND LIB_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/cutlass/gemm/bladnn_per_channel.cu)
  list(APPEND LIB_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/cutlass/conv/bladnn_per_channel.cu)
  list(APPEND LIB_SOURCES ${LIB_CUSTOM_CUTLASS_SOURCES})
  include_directories(
    include
    third_party/cutlass/include
    cutlass_library/tools/library/include
    cutlass_library/tools/util/include
    /usr/local/cuda/include)
endif()

# Generate the static library from the library sources
add_library(blade_gemm
  STATIC
  ${LIB_SOURCES})

if(USE_TVM) 
  add_dependencies(blade_gemm json_extern)
endif()

set_target_properties(blade_gemm PROPERTIES PUBLIC_HEADER "include/bladnn/bladnn.h")

if(NOT USE_TVM)
  target_link_libraries(blade_gemm
    cutlass_library_objs cudnn cublasLt)

  add_executable(cuda_conv2d examples/cuda_conv2d.cc)
  target_link_libraries(cuda_conv2d
    blade_gemm cutlass_library_objs ${CUDA_LIBRARIES}
  )

  add_executable(cuda_conv2d_int8 examples/cuda_conv2d_int8.cc)
  target_link_libraries(cuda_conv2d_int8
    blade_gemm cutlass_library_objs ${CUDA_LIBRARIES}
  )

  add_executable(cuda_spgemm examples/cuda_spgemm.cc)
  target_link_libraries(cuda_spgemm
    blade_gemm cutlass_library_objs ${CUDA_LIBRARIES}
  )

  add_executable(cuda_gemm examples/cuda_gemm.cc)
  target_link_libraries(cuda_gemm
    blade_gemm cutlass_library_objs ${CUDA_LIBRARIES}
  )

  add_executable(cuda_gemm_int8 examples/cuda_gemm_int8.cc)
  target_link_libraries(cuda_gemm_int8
    blade_gemm cutlass_library_objs ${CUDA_LIBRARIES}
  )

  # for per-channel quant example
  set ( PRJ_COMPILE_OPTIONS )
  enable_language( CUDA )
    list ( APPEND PRJ_COMPILE_OPTIONS -Xcompiler )

  # for per-channel quant example

endif()

install(TARGETS blade_gemm
  LIBRARY DESTINATION lib64
  ARCHIVE DESTINATION lib64
  PUBLIC_HEADER DESTINATION include/bladnn)
