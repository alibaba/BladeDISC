load("//bazel:common.bzl", "get_python_bin", "get_env_bool_value_str")

_PYTHON_BIN_PATH = "PYTHON_BIN_PATH"
_BLADE_NEED_TENSORRT = "BLADE_WITH_TENSORRT"
_BLADE_BUILD_INTERNAL = "BLADE_WITH_INTERNAL"

def _blade_helper_impl(repository_ctx):
    repository_ctx.template("build_defs.bzl", Label("//bazel/blade_helper:build_defs.bzl.tpl"), {
        "%{PYTHON_BIN_PATH}": get_python_bin(repository_ctx),
        "%{TENSORRT_ENABLED}": get_env_bool_value_str(repository_ctx, _BLADE_NEED_TENSORRT),
        "%{IF_INTERNAL}": get_env_bool_value_str(repository_ctx, _BLADE_BUILD_INTERNAL),
    })

    repository_ctx.template("BUILD", Label("//bazel/blade_helper:BUILD.tpl"), {
    })

blade_helper_configure = repository_rule(
    implementation = _blade_helper_impl,
    environ = [
        _PYTHON_BIN_PATH,
        _BLADE_NEED_TENSORRT,
        _BLADE_BUILD_INTERNAL,
    ],
)
