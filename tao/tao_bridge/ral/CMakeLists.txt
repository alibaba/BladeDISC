list(APPEND RAL_PROTO_FILES
    "tensorflow/compiler/mlir/xla/compile_metadata.proto"
)

add_definitions("-DDISC_BUILD_FROM_TF_BRIDGE")

if(NOT ${TAO_CPU_ONLY})
  add_definitions("-DTAO_RAL_USE_STREAM_EXECUTOR")
endif()

# generate protobuf file
foreach(file ${RAL_PROTO_FILES})
    message("Generating proto: ${file}")
    EXECUTE_PROCESS(
        COMMAND ${Protobuf_PROTOC_EXECUTABLE} --cpp_out=. ${file}
        WORKING_DIRECTORY ${tao_ops_SOURCE_DIR}/tao_bridge/ral
        RESULT_VARIABLE CMD_RESULT
        OUTPUT_VARIABLE STDOUT
        ERROR_VARIABLE STDERR
    )
    if (NOT CMD_RESULT EQUAL "0")
        message(FATAL_ERROR "Generating proto failed: \n${STDOUT}\n${STDERR}")
    endif()
endforeach()

list(APPEND RAL_NEED_RENAME_PROTO_FILES
  "tensorflow/compiler/mlir/xla/compile_metadata.pb.h"
  "tensorflow/compiler/mlir/xla/compile_metadata.pb.cc"
)

if (Tensorflow_USE_PB3)
    MESSAGE("Renaming namespace from google::protobuf to google::protobuf3.")
    EXECUTE_PROCESS(
        COMMAND sed -i "s#::protobuf::#::protobuf3::#g; s#namespace protobuf #namespace protobuf3 #g" ${RAL_NEED_RENAME_PROTO_FILES}
        WORKING_DIRECTORY ${tao_ops_SOURCE_DIR}/tao_bridge/ral
    )
    add_compile_options(-DRAL_NEED_PROTO3)
endif()

if(${TAO_CUDA})
    ############### cuda config ###############
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    if(NOT ${CUDA_VERSION_STRING} VERSION_LESS "11.0")
      set(RAL_GPU_ARCHS "60;61;70;72;75;80" CACHE STRING "GPU SM")
      set(CMAKE_CUDA_STANDARD 14)
    elseif(NOT ${CUDA_VERSION_STRING} VERSION_LESS "10.0")
      set(RAL_GPU_ARCHS "60;61;70;72;75" CACHE STRING "GPU SM")
    elseif(NOT ${CUDA_VERSION_STRING} VERSION_LESS "9.1")
      set(RAL_GPU_ARCHS "60;61;70;72" CACHE STRING "GPU SM")
    else()
      set(RAL_GPU_ARCHS "60;61;70" CACHE STRING "GPU SM")
    endif()

    foreach(arch ${RAL_GPU_ARCHS})
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${arch},code=sm_${arch}")
    endforeach()
    list(GET RAL_GPU_ARCHS -1 ptx)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${ptx},code=compute_${ptx}")
    include_directories("/usr/local/cuda/targets/x86_64-linux/include")
endif()

if(${TAO_ROCM} OR ${TAO_DCU})
    set(CMAKE_HIP_STANDARD 11)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    MESSAGE("Include header files from ROCM path ${ROCM_PATH}")
    include_directories("${ROCM_PATH}/include")
    include_directories("${ROCM_PATH}/include/rocrand")
    include_directories("${ROCM_PATH}/include/rocprim")
    include_directories("${ROCM_PATH}/include/rocprofiler")
    include_directories("${ROCM_PATH}/include/roctracer")
    add_definitions(-D__HIP_PLATFORM_HCC__ -DTENSORFLOW_USE_ROCM) 
endif()


list(APPEND RAL_HDRS
    "tensorflow/compiler/mlir/xla/compile_metadata.pb.h"
    "tensorflow/compiler/mlir/xla/ral/context/context_util.h"
    "tensorflow/compiler/mlir/xla/ral/context/common_context_impl.h"
    "tensorflow/compiler/mlir/xla/ral/context/stream_executor_based_impl.h"
    "tensorflow/compiler/mlir/xla/ral/context/tensorflow/tf_context_impl.h"
    "tensorflow/compiler/mlir/xla/ral/device/cpu/cpu_driver.h"
    "tensorflow/compiler/mlir/xla/ral/device/gpu/gpu_driver.h"
    "tensorflow/compiler/mlir/xla/ral/ral_api.h"
    "tensorflow/compiler/mlir/xla/ral/ral_base.h"
    "tensorflow/compiler/mlir/xla/ral/ral_context.h"
    "tensorflow/compiler/mlir/xla/ral/ral_driver.h"
    "tensorflow/compiler/mlir/xla/ral/ral_helper.h"
    "tensorflow/compiler/mlir/xla/ral/ral_logging.h"
)

list(APPEND RAL_SRCS
    "tensorflow/compiler/mlir/xla/compile_metadata.pb.cc"
    "tensorflow/compiler/mlir/xla/ral/context/common_context_impl.cc"
    "tensorflow/compiler/mlir/xla/ral/context/stream_executor_based_impl.cc"
    "tensorflow/compiler/mlir/xla/ral/context/tensorflow/tf_context_impl.cc"
    "tensorflow/compiler/mlir/xla/ral/context/tensorflow/tf_kernel_impl.cc"
    "tensorflow/compiler/mlir/xla/ral/device/cpu/cpu_driver.cc"
    "tensorflow/compiler/mlir/xla/ral/device/gpu/gpu_driver.cc"
    "tensorflow/compiler/mlir/xla/ral/ral_api.cc"
    "tensorflow/compiler/mlir/xla/ral/ral_context.cc"
    "tensorflow/compiler/mlir/xla/ral/ral_helper.cc"
    "tensorflow/compiler/mlir/xla/ral/ral_logging.cc"
)

#TODO: revisit this when support DCU in tf bridge
if(${TAO_CUDA})
  list(APPEND RAL_SRCS
      "tensorflow/compiler/mlir/xla/ral/context/dynamic_sort_impl.cc"
      "tensorflow/compiler/mlir/xla/ral/context/random_impl.cc"
      "tensorflow/compiler/mlir/xla/ral/context/common_context_impl_cuda.cc"
  )
  list(APPEND RAL_CU_SRCS
      "tensorflow/compiler/mlir/xla/ral/context/custom_library/dynamic_sort.cu.cc"
      "tensorflow/compiler/mlir/xla/ral/context/custom_library/random_gpu.cu.cc"
  )

  foreach(file ${RAL_CU_SRCS})
    set_source_files_properties(${file} PROPERTIES LANGUAGE CUDA)
  endforeach()

  include_directories("${CMAKE_SOURCE_DIR}")
  include_directories("${CMAKE_SOURCE_DIR}/third_party/cub")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()

if(${TAO_ROCM} OR ${TAO_DCU})
  list(APPEND RAL_SRCS
    "tensorflow/compiler/mlir/xla/ral/context/common_context_impl_cuda.cc"
  )
  include_directories("${CMAKE_SOURCE_DIR}")
  list(APPEND RAL_HDRS
    "tensorflow/compiler/mlir/xla/ral/context/common_context_impl_cuda.h"
  )
endif()

#CustomLib Support for DCU to be revisited later
if(${TAO_ROCM})
  list(APPEND RAL_SRCS
  "tensorflow/compiler/mlir/xla/ral/context/dynamic_sort_impl.cc"
  "tensorflow/compiler/mlir/xla/ral/context/random_impl.cc"
  )
  list(APPEND RAL_CU_SRCS
    "tensorflow/compiler/mlir/xla/ral/context/custom_library/dynamic_sort.cu.cc"
    "tensorflow/compiler/mlir/xla/ral/context/custom_library/random_gpu.cu.cc"
  )
  foreach(file ${RAL_CU_SRCS})
  set_source_files_properties(${file} PROPERTIES LANGUAGE HIP)
  endforeach()
endif()

if(${TAO_ENABLE_MKLDNN})
  list(APPEND RAL_SRCS
    "tensorflow/compiler/mlir/xla/ral/context/common_context_impl_mkldnn.cc"
  )
endif()

if(${TAO_CPU_ONLY})
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

include_directories(".")

add_library(ral OBJECT ${RAL_SRCS} ${RAL_CU_SRCS})

if(${ENABLE_BLADE_GEMM})
  include_directories(${PLATFORM_ALIBABA_DIR}/blade_gemm/include)
  target_link_libraries(ral PRIVATE blade_gemm)
endif()

if(${TAO_ENABLE_MKLDNN})
  target_link_libraries(ral PRIVATE dnnl)
  if(${TAO_X86})
    set(MKL_LINK_OPTION "-Wl,--start-group ${MKL_ROOT}/lib/libmkl_intel_ilp64.a ${MKL_ROOT}/lib/libmkl_gnu_thread.a ${MKL_ROOT}/lib/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl")
    set(MKL_COMPILE_OPTION -DMKL_ILP64 -m64 -I${MKL_ROOT}/include)
    target_compile_options(ral PRIVATE ${MKL_COMPILE_OPTION})
    target_link_libraries(ral PRIVATE ${MKL_LINK_OPTION})
  endif()
endif()
