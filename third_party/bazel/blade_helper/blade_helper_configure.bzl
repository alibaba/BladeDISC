load("//bazel:common.bzl", "get_python_bin", "get_env_bool_value_str", "get_host_environ")

_PYTHON_BIN_PATH = "PYTHON_BIN_PATH"
_BLADE_NEED_TENSORRT = "BLADE_WITH_TENSORRT"
_BLADE_BUILD_INTERNAL = "BLADE_WITH_INTERNAL"
_CC_BIN_PATH = "CC"
_CXX_BIN_PATH = "CXX"
_NVCC_BIN_PATH = "NVCC"
_CUDA_HOME = "TF_CUDA_HOME"
_CUDA_VERSION = "TF_CUDA_VERSION"
_IF_HIE = "BLADE_WITH_HIE"

def _blade_helper_impl(repository_ctx):
    repository_ctx.template("build_defs.bzl", Label("//bazel/blade_helper:build_defs.bzl.tpl"), {
        "%{PYTHON_BIN_PATH}": get_python_bin(repository_ctx),
        "%{TENSORRT_ENABLED}": get_env_bool_value_str(repository_ctx, _BLADE_NEED_TENSORRT),
        "%{IF_INTERNAL}": get_env_bool_value_str(repository_ctx, _BLADE_BUILD_INTERNAL),
        "%{CC_BIN_PATH}": get_host_environ(repository_ctx, _CC_BIN_PATH, ""),
        "%{CXX_BIN_PATH}": get_host_environ(repository_ctx, _CXX_BIN_PATH, ""),
        "%{NVCC_BIN_PATH}": get_host_environ(repository_ctx, _NVCC_BIN_PATH, ""),
        "%{CUDA_HOME}": get_host_environ(repository_ctx, _CUDA_HOME, ""),
        "%{CUDA_VERSION}": get_host_environ(repository_ctx, _CUDA_VERSION, ""),
        "%{IF_HIE}": get_env_bool_value_str(repository_ctx, _IF_HIE),
    })

    repository_ctx.template("BUILD", Label("//bazel/blade_helper:BUILD.tpl"), {
    })

blade_helper_configure = repository_rule(
    implementation = _blade_helper_impl,
    environ = [
        _PYTHON_BIN_PATH,
        _BLADE_NEED_TENSORRT,
        _BLADE_BUILD_INTERNAL,
        _CC_BIN_PATH,
        _CXX_BIN_PATH,
        _NVCC_BIN_PATH,
        _CUDA_HOME,
        _CUDA_VERSION,
        _IF_HIE,
    ],
)
