load("//bazel:common.bzl", "get_python_bin")

_PYTHON_BIN_PATH = "PYTHON_BIN_PATH"

def _blade_helper_impl(repository_ctx):
    py_bin = get_python_bin(repository_ctx)
    repository_ctx.template("build_defs.bzl", Label("//bazel/blade_helper:build_defs.bzl.tpl"), {
        "%{PYTHON_BIN_PATH}": py_bin,
    })

    repository_ctx.template("BUILD", Label("//bazel/blade_helper:BUILD.tpl"), {
    })

blade_helper_configure = repository_rule(
    implementation = _blade_helper_impl,
    environ = [
        _PYTHON_BIN_PATH,
    ],
)
