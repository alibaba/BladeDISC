def python_bin_path():
    return "%{PYTHON_BIN_PATH}"

def cc_bin_path():
    return "%{CC_BIN_PATH}"

def cxx_bin_path():
    return "%{CXX_BIN_PATH}"

def nvcc_bin_path():
    return "%{NVCC_BIN_PATH}"

def blade_gemm_nvcc():
    return "%{BLADE_GEMM_NVCC}"

def blade_gemm_tvm():
    return "%{BLADE_GEMM_TVM}"

def blade_gemm_rocm_path():
    return "%{BLADE_GEMM_ROCM_PATH}"

def blade_gemm_nvcc_archs():
    return "%{BLADE_GEMM_NVCC_ARCHS}"

def blade_gemm_library_kernels():
    return "%{BLADE_GEMM_LIBRARY_KERNELS}"

def cuda_home():
    return "%{CUDA_HOME}"

def cuda_version():
    return "%{CUDA_VERSION}"

def disc_target_cpu_arch():
    return "%{DISC_TARGET_CPU_ARCH}"

def if_hie_enabled(if_true, if_false = []):
    if %{IF_HIE}:
        return if_true
    return if_false

def if_hie_disabled(if_true, if_false = []):
    if not %{IF_HIE}:
        return if_true
    return if_false

def if_tensorrt_enabled(if_true, if_false = []):
    if %{TENSORRT_ENABLED}:
        return if_true
    return if_false

def if_tensorrt_disabled(if_true, if_false = []):
    if not %{TENSORRT_ENABLED}:
        return if_true
    return if_false

def disc_build_version():
    return "%{DISC_BUILD_VERSION}"

def disc_build_git_branch():
    return "%{DISC_BUILD_GIT_BRANCH}"

def disc_build_git_head():
    return "%{DISC_BUILD_GIT_HEAD}"

def disc_build_host():
    return "%{DISC_BUILD_HOST}"

def disc_build_ip():
    return "%{DISC_BUILD_IP}"

def disc_build_time():
    return "%{DISC_BUILD_TIME}"

def if_disc_mkldnn(if_true, if_false=[]):
    return select({
        "@local_config_blade_disc_helper//:is_mkldnn": if_true,
        "//conditions:default": if_false
    })

def if_disc_aarch64(if_true, if_false=[]):
    return select({
        "@local_config_blade_disc_helper//:disc_aarch64": if_true,
        "//conditions:default": if_false
    })

def if_platform_alibaba(if_true, if_false=[]):
    return select({
        "@local_config_blade_disc_helper//:is_platform_alibaba": if_true,
        "//conditions:default": if_false
    })

def foreign_make_args():
    return [ "-j%{DISC_FOREIGN_MAKE_JOBS}" ]

def if_internal_serving(if_true, if_false=[]):
    return select({
        "@local_config_blade_disc_helper//:is_internal_serving": if_true,
        "//conditions:default": if_false
    })
