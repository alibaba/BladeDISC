load("//bazel:common.bzl", "get_python_bin", "get_host_environ")

_PYTHON_BIN_PATH = "PYTHON_BIN_PATH"
_BLADE_NEED_TENSORRT = "BLADE_WITH_TENSORRT"
_BLADE_BUILD_INTERNAL = "BLADE_WITH_INTERNAL"

def _blade_helper_impl(repository_ctx):
    py_bin = get_python_bin(repository_ctx)
    tensorrt_enabled = "True" if get_host_environ(repository_ctx, _BLADE_NEED_TENSORRT, "False").lower() in ["1", "true", "on"] else "False"
    is_internal = "True" if get_host_environ(repository_ctx, _BLADE_BUILD_INTERNAL, "False").lower() in ["1", "true", "on"] else "False"
    repository_ctx.template("build_defs.bzl", Label("//bazel/blade_helper:build_defs.bzl.tpl"), {
        "%{PYTHON_BIN_PATH}": py_bin,
        "%{TENSORRT_ENABLED}": tensorrt_enabled,
        "%{IF_INTERNAL}": is_internal,
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
