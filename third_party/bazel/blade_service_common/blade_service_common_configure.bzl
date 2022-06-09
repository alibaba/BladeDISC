load("//bazel:common.bzl", "get_env_bool_value")

_IS_PLATFORM_ALIBABA = "IS_PLATFORM_ALIBABA"

def _blade_service_common_impl(repository_ctx):
    if _IS_PLATFORM_ALIBABA not in repository_ctx.os.environ or not get_env_bool_value(repository_ctx, _IS_PLATFORM_ALIBABA):
        repository_ctx.template("blade_service_common_workspace.bzl", Label("//bazel/blade_service_common:blade_service_common_empty_workspace.bzl.tpl"), {
        })
    elif _IS_PLATFORM_ALIBABA in repository_ctx.os.environ and get_env_bool_value(repository_ctx, _IS_PLATFORM_ALIBABA):
        repository_ctx.template("blade_service_common_workspace.bzl", Label("//bazel/blade_service_common:blade_service_common_workspace.bzl.tpl"), {
        })

    repository_ctx.template("BUILD", Label("//bazel/blade_service_common:BUILD.tpl"), {
    })

blade_service_common_configure = repository_rule(
    implementation = _blade_service_common_impl,
    environ = [
        _IS_PLATFORM_ALIBABA,
    ],
)
