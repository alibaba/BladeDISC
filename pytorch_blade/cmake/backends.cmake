resolve_env(TORCH_BLADE_BUILD_TENSORRT ON)

if(TORCH_BLADE_BUILD_TENSORRT)
  include(cmake/tensorrt.cmake)
endif(TORCH_BLADE_BUILD_TENSORRT)

