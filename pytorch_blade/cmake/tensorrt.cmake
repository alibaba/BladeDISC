resolve_env(TORCH_BLADE_STATIC_LINK_TENSORRT OFF)
# TENSORRT_INSTALL_PATH
resolve_env(TENSORRT_INSTALL_PATH)
# CUDA_HOME
resolve_env(CUDA_HOME)
# TENSORRT_CUDNN_HOME
resolve_env(TENSORRT_CUDNN_HOME ${CUDA_HOME})

file(GLOB_RECURSE TORCH_BLADE_TENSORRT_BRIDGE_HEADERS src/compiler/tensorrt/bridge/*.h)
file(GLOB_RECURSE TORCH_BLADE_TENSORRT_BRIDGE_SRCS src/compiler/tensorrt/bridge/*.cpp ${TORCH_BLADE_TENSORRT_BRIDGE_HEADERS})

# TensorRT bridge library was introduced to decouple TensorRT with Torch.
# Because TensorRT & Torch maybe linked with different CuDNN version, which
# would trigger TensorRT runtime error.
#
# To address the problem we static linked TensorRT & CuDNN into
# torch_blade_tensorrt_bridge library and hide all symbols that need not be exported.
# TODO: Here, STATIC library leads CuDNN version conflicts.
add_library(torch_blade_tensorrt_bridge SHARED ${TORCH_BLADE_TENSORRT_BRIDGE_SRCS})
target_include_directories(torch_blade_tensorrt_bridge PRIVATE
  ${TENSORRT_INSTALL_PATH}/include
  ${CUDA_HOME}/include
  ${PYTORCH_DIR}/include
  src/
)

target_link_directories(torch_blade_tensorrt_bridge PRIVATE
  ${TENSORRT_INSTALL_PATH}/lib
  ${TENSORRT_CUDNN_HOME}/lib64
  ${CUDA_HOME}/lib64
  ${CUDA_HOME}/lib
)

if (TORCH_BLADE_STATIC_LINK_TENSORRT)
  message(STATUS "Use static link to tensorrt")

  list(APPEND TORCH_BLADE_CUDA_STATIC_LIBRARIES ${CUDA_HOME}/lib64/libculibos.a)
  list(APPEND TORCH_BLADE_CUDA_STATIC_LIBRARIES ${CUDA_HOME}/lib64/libcublas_static.a)
  if (TORCH_BLADE_CUDA_VERSION GREATER "10.0")
     list(APPEND TORCH_BLADE_CUDA_STATIC_LIBRARIES ${CUDA_HOME}/lib64/libcublasLt_static.a)
  endif()

  if(EXISTS ${TENSORRT_INSTALL_PATH}/lib/libmyelin_compiler_static.a)
     # These myelin libraries was standalone in TensorRT 7
     list(APPEND TENSORRT_MYELIN_STATIC_LIBRARIES
         ${TENSORRT_INSTALL_PATH}/lib/libmyelin_compiler_static.a
         ${TENSORRT_INSTALL_PATH}/lib/libmyelin_executor_static.a
         ${TENSORRT_INSTALL_PATH}/lib/libmyelin_pattern_library_static.a
         ${TENSORRT_INSTALL_PATH}/lib/libmyelin_pattern_runtime_static.a)
  endif()

  # static link & hide tensorrt, cudnn
  target_link_libraries(torch_blade_tensorrt_bridge PRIVATE
    ${TENSORRT_INSTALL_PATH}/lib/libnvonnxparser_static.a
    ${TENSORRT_INSTALL_PATH}/lib/libonnx_proto.a
    ${TENSORRT_INSTALL_PATH}/lib/libprotobuf.a
  # Note:
  # Link to TensorRT static library may introduce performance decay.
  # We try to fix this issue, according to suggestions from
  # https://docs.nvidia.com/deeplearning/tensorrt/sample-support-guide/index.html#building-samples
  -Wl,-whole-archive
    ${TENSORRT_INSTALL_PATH}/lib/libnvinfer_plugin_static.a
    ${TENSORRT_INSTALL_PATH}/lib/libnvinfer_static.a
  -Wl,-no-whole-archive
    ${TENSORRT_MYELIN_STATIC_LIBRARIES}
    ${TENSORRT_CUDNN_HOME}/lib64/libcudnn_static.a
    ${TORCH_BLADE_CUDA_STATIC_LIBRARIES}
    nvrtc dl rt)

  target_link_options(torch_blade_tensorrt_bridge PRIVATE "-Wl,--exclude-libs,ALL")
else()
  target_link_libraries(torch_blade_tensorrt_bridge PRIVATE
    nvonnxparser nvinfer nvinfer_plugin)
endif()


file(GLOB TORCH_BLADE_TENSORRT_TEST_SRCS src/compiler/tensorrt/*test.cpp)
file(GLOB TORCH_BLADE_TENSORRT_PYTHON_SRCS src/compiler/tensorrt/*pybind*.cpp src/compiler/tensorrt/*pybind*.h)
file(GLOB TORCH_BLADE_TENSORRT_SRCS src/compiler/tensorrt/*.cpp src/compiler/tensorrt/*.h)
exclude(TORCH_BLADE_TENSORRT_SRCS "${TORCH_BLADE_TENSORRT_SRCS}" "${TORCH_BLADE_TENSORRT_TEST_SRCS}" "${TORCH_BLADE_TENSORRT_PYTHON_SRCS}")

add_library(torch_blade_tensorrt INTERFACE)
target_sources(torch_blade_tensorrt INTERFACE ${TORCH_BLADE_TENSORRT_SRCS})
# TODO: Eventually, we should not depend on TensorRT headers here.
# But we need some extra works to make it clean.
target_include_directories(torch_blade_tensorrt INTERFACE
  ${TENSORRT_INSTALL_PATH}/include
)
target_link_libraries(torch_blade_tensorrt INTERFACE torch_blade_tensorrt_bridge)
target_compile_definitions(torch_blade PUBLIC -DTORCH_BLADE_BUILD_TENSORRT)
target_link_libraries(torch_blade PRIVATE torch_blade_tensorrt)

if (TORCH_BLADE_BUILD_PYTHON_SUPPORT)
  add_library(_torch_blade_tensorrt INTERFACE)
  # TODO: Eventually, we should not depend on TensorRT headers here.
  # But we need some extra works to make it clean.
  target_include_directories(_torch_blade_tensorrt INTERFACE
    ${TENSORRT_INSTALL_PATH}/include
  )
  target_sources(_torch_blade_tensorrt INTERFACE ${TORCH_BLADE_TENSORRT_PYTHON_SRCS})
  target_compile_definitions(_torch_blade PUBLIC -DTORCH_BLADE_BUILD_TENSORRT)
  target_link_libraries(_torch_blade PRIVATE _torch_blade_tensorrt)
endif (TORCH_BLADE_BUILD_PYTHON_SUPPORT)

install(TARGETS torch_blade_tensorrt_bridge LIBRARY PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE)
