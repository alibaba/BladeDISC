# FindTorch
# -------
#
# Finds the Torch library
#
# This will define the following variables:
#
#   TORCH_FOUND        -- True if the system has the Torch library
#   TORCH_INCLUDE_DIRS -- The include directories for torch
#   TORCH_LIBRARIES    -- Libraries to link against
#   TORCH_CXX_FLAGS    -- Additional (required) compiler flags
#
# and the following imported targets:
#
#   torch

include(FindPackageHandleStandardArgs)
set(TORCH_INSTALL_PREFIX ${PYTORCH_DIR})
message("TORCH_INSTALL_PREFIX=${TORCH_INSTALL_PREFIX}")

# Include directories.
if (EXISTS "${TORCH_INSTALL_PREFIX}/include")
  set(TORCH_INCLUDE_DIRS
    ${TORCH_INSTALL_PREFIX}/include
    ${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include)
else()
  set(TORCH_INCLUDE_DIRS
    ${TORCH_INSTALL_PREFIX}/include
    ${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include)
endif()

# Library dependencies.
add_library(torch SHARED IMPORTED)
find_library(TORCH_LIBRARY torch PATHS "${TORCH_INSTALL_PREFIX}/lib")
set(TORCH_LIBRARIES torch)

find_library(TORCH_CPU_LIBRARY torch_cpu PATHS "${TORCH_INSTALL_PREFIX}/lib")
list(APPEND TORCH_LIBRARIES ${TORCH_CPU_LIBRARY})

find_library(C10_LIBRARY c10 PATHS "${TORCH_INSTALL_PREFIX}/lib")
list(APPEND TORCH_LIBRARIES ${C10_LIBRARY})

if (TORCH_CUDA)
  if (TORCH_ROCM)
    find_library(TORCH_HIP_LIBRARY torch_hip PATHS "${TORCH_INSTALL_PREFIX}/lib")
    list(APPEND TORCH_LIBRARIES ${TORCH_HIP_LIBRARY})
    find_library(C10_HIP_LIBRARY c10_hip PATHS "${TORCH_INSTALL_PREFIX}/lib")
    list(APPEND TORCH_LIBRARIES ${C10_HIP_LIBRARY})
  else (TORCH_ROCM)
    find_library(TORCH_CUDA_LIBRARY torch_cuda PATHS "${TORCH_INSTALL_PREFIX}/lib")
    list(APPEND TORCH_LIBRARIES ${TORCH_CUDA_LIBRARY})
    find_library(C10_CUDA_LIBRARY c10_cuda PATHS "${TORCH_INSTALL_PREFIX}/lib")
    list(APPEND TORCH_LIBRARIES ${C10_CUDA_LIBRARY})
  endif (TORCH_ROCM)
endif (TORCH_CUDA)


set_target_properties(torch PROPERTIES
    IMPORTED_LOCATION "${TORCH_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIRS}"
    CXX_STANDARD 11
)
if (TORCH_CXX_FLAGS)
  set_target_properties(torch PROPERTIES INTERFACE_COMPILE_OPTIONS "${TORCH_CXX_FLAGS}")
endif()

find_package_handle_standard_args(Torch DEFAULT_MSG TORCH_LIBRARY TORCH_INCLUDE_DIRS)
