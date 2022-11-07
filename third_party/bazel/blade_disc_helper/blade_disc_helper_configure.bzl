load("//bazel:common.bzl", "get_python_bin", "get_env_bool_value_str", "get_host_environ")

_PYTHON_BIN_PATH = "PYTHON_BIN_PATH"
_BLADE_NEED_TENSORRT = "BLADE_WITH_TENSORRT"
_CC_BIN_PATH = "CC"
_CXX_BIN_PATH = "CXX"
_NVCC_BIN_PATH = "NVCC"
_CUDA_HOME = "TF_CUDA_HOME"
_CUDA_VERSION = "TF_CUDA_VERSION"
_IF_HIE = "BLADE_WITH_HIE"
# for generating disc compiler's version.h
_DISC_BUILD_VERSION = "DISC_BUILD_VERSION"
_DISC_BUILD_GIT_BRANCH = "DISC_BUILD_GIT_BRANCH"
_DISC_BUILD_GIT_HEAD = "DISC_BUILD_GIT_HEAD"
_DISC_BUILD_HOST = "DISC_BUILD_HOST"
_DISC_BUILD_IP = "DISC_BUILD_IP"
_DISC_BUILD_TIME = "DISC_BUILD_TIME"
_BLADE_GEMM_NVCC = "BLADE_GEMM_NVCC"
_BLADE_GEMM_NVCC_ARCHS = "BLADE_GEMM_NVCC_ARCHS"
_BLADE_GEMM_LIBRARY_KERNELS = "BLADE_GEMM_LIBRARY_KERNELS"
_BLADE_GEMM_TVM = "BLADE_GEMM_TVM"
_BLADE_GEMM_ROCM_PATH = "BLADE_GEMM_ROCM_PATH"
_DISC_TARGET_CPU_ARCH = "DISC_TARGET_CPU_ARCH"
_DISC_FOREIGN_MAKE_JOBS = "DISC_FOREIGN_MAKE_JOBS"

def _blade_disc_helper_impl(repository_ctx):
    repository_ctx.template("build_defs.bzl", Label("//bazel/blade_disc_helper:build_defs.bzl.tpl"), {
        "%{PYTHON_BIN_PATH}": get_python_bin(repository_ctx),
        "%{TENSORRT_ENABLED}": get_env_bool_value_str(repository_ctx, _BLADE_NEED_TENSORRT),
        "%{IF_HIE}": get_env_bool_value_str(repository_ctx, _IF_HIE),
        "%{CC_BIN_PATH}": get_host_environ(repository_ctx, _CC_BIN_PATH, ""),
        "%{CXX_BIN_PATH}": get_host_environ(repository_ctx, _CXX_BIN_PATH, ""),
        "%{NVCC_BIN_PATH}": get_host_environ(repository_ctx, _NVCC_BIN_PATH, ""),
        "%{CUDA_HOME}": get_host_environ(repository_ctx, _CUDA_HOME, ""),
        "%{CUDA_VERSION}": get_host_environ(repository_ctx, _CUDA_VERSION, ""),
        "%{DISC_BUILD_VERSION}": get_host_environ(repository_ctx, _DISC_BUILD_VERSION, ""),
        "%{DISC_BUILD_GIT_BRANCH}": get_host_environ(repository_ctx, _DISC_BUILD_GIT_BRANCH, ""),
        "%{DISC_BUILD_GIT_HEAD}": get_host_environ(repository_ctx, _DISC_BUILD_GIT_HEAD, ""),
        "%{DISC_BUILD_HOST}": get_host_environ(repository_ctx, _DISC_BUILD_HOST, ""),
        "%{DISC_BUILD_IP}": get_host_environ(repository_ctx, _DISC_BUILD_IP, ""),
        "%{DISC_BUILD_TIME}": get_host_environ(repository_ctx, _DISC_BUILD_TIME, ""),
        "%{BLADE_GEMM_NVCC}": get_host_environ(repository_ctx, _BLADE_GEMM_NVCC, ""),
        "%{BLADE_GEMM_TVM}": get_host_environ(repository_ctx, _BLADE_GEMM_TVM, ""),
        "%{BLADE_GEMM_ROCM_PATH}": get_host_environ(repository_ctx, _BLADE_GEMM_ROCM_PATH, ""),
        "%{DISC_TARGET_CPU_ARCH}": get_host_environ(repository_ctx, _DISC_TARGET_CPU_ARCH, ""),
        "%{DISC_FOREIGN_MAKE_JOBS}": get_host_environ(repository_ctx, _DISC_FOREIGN_MAKE_JOBS, ""),
    })

    repository_ctx.template("BUILD", Label("//bazel/blade_disc_helper:BUILD.tpl"), {
    })

blade_disc_helper_configure = repository_rule(
    implementation = _blade_disc_helper_impl,
    environ = [
        _PYTHON_BIN_PATH,
        _BLADE_NEED_TENSORRT,
        _CC_BIN_PATH,
        _CXX_BIN_PATH,
        _NVCC_BIN_PATH,
        _CUDA_HOME,
        _CUDA_VERSION,
        _IF_HIE,
        _DISC_BUILD_VERSION,
        _DISC_BUILD_GIT_BRANCH,
        _DISC_BUILD_GIT_HEAD,
        _DISC_BUILD_HOST,
        _DISC_BUILD_IP,
        _DISC_BUILD_TIME,
        _BLADE_GEMM_NVCC,
        _BLADE_GEMM_NVCC_ARCHS,
        _BLADE_GEMM_LIBRARY_KERNELS,
        _BLADE_GEMM_TVM,
        _BLADE_GEMM_ROCM_PATH,
        _DISC_TARGET_CPU_ARCH,
        _DISC_FOREIGN_MAKE_JOBS,
    ],
)
