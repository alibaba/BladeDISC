load("//bazel:common.bzl", "get_python_bin", "get_env_bool_value_str", "get_host_environ")

_PYTHON_BIN_PATH = "PYTHON_BIN_PATH"
_BLADE_NEED_TENSORRT = "BLADE_WITH_TENSORRT"
_BLADE_BUILD_INTERNAL = "BLADE_WITH_INTERNAL"
_TAO_BUILD_VERSION = "TAO_BUILD_VERSION"
_TAO_BUILD_GIT_BRANCH = "TAO_BUILD_GIT_BRANCH"
_TAO_BUILD_GIT_HEAD = "TAO_BUILD_GIT_HEAD"
_TAO_BUILD_HOST = "TAO_BUILD_HOST"
_TAO_BUILD_IP = "TAO_BUILD_IP"
_TAO_BUILD_TIME = "TAO_BUILD_TIME"


def _blade_disc_helper_impl(repository_ctx):
    repository_ctx.template("build_defs.bzl", Label("//bazel/blade_disc_helper:build_defs.bzl.tpl"), {
        "%{PYTHON_BIN_PATH}": get_python_bin(repository_ctx),
        "%{TENSORRT_ENABLED}": get_env_bool_value_str(repository_ctx, _BLADE_NEED_TENSORRT),
        "%{IF_INTERNAL}": get_env_bool_value_str(repository_ctx, _BLADE_BUILD_INTERNAL),
        "%{TAO_BUILD_VERSION}": get_host_environ(repository_ctx, _TAO_BUILD_VERSION),
        "%{TAO_BUILD_GIT_BRANCH}": get_host_environ(repository_ctx, _TAO_BUILD_GIT_BRANCH),
        "%{TAO_BUILD_GIT_HEAD}": get_host_environ(repository_ctx, _TAO_BUILD_GIT_HEAD),
        "%{TAO_BUILD_HOST}": get_host_environ(repository_ctx, _TAO_BUILD_HOST),
        "%{TAO_BUILD_IP}": get_host_environ(repository_ctx, _TAO_BUILD_IP),
        "%{TAO_BUILD_TIME}": get_host_environ(repository_ctx, _TAO_BUILD_TIME),
    })

    repository_ctx.template("BUILD", Label("//bazel/blade_disc_helper:BUILD.tpl"), {
    })


blade_disc_helper_configure = repository_rule(
    implementation = _blade_disc_helper_impl,
    environ = [
        _PYTHON_BIN_PATH,
        _BLADE_NEED_TENSORRT,
        _BLADE_BUILD_INTERNAL,
        _TAO_BUILD_VERSION,
        _TAO_BUILD_GIT_BRANCH,
        _TAO_BUILD_GIT_HEAD,
        _TAO_BUILD_HOST,
        _TAO_BUILD_IP,
        _TAO_BUILD_TIME
    ],
)
