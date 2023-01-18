load("//bazel:common.bzl", "get_env_bool_value")

_IS_PLATFORM_ALIBABA = "IS_PLATFORM_ALIBABA"
_IS_INTERNAL_SERVING = "IS_INTERNAL_SERVING"

def _tpl(repository_ctx, tpl, substitutions):
    repository_ctx.template(
        tpl,
        Label("//bazel/blade_service_common:%s.tpl" % tpl),
        substitutions,
    )

def _blade_service_common_impl(repository_ctx):
    if get_env_bool_value(repository_ctx, _IS_PLATFORM_ALIBABA):
        if get_env_bool_value(repository_ctx, _IS_INTERNAL_SERVING):
            substitutions = {
                "%{RULES_FOREIGN_CC_MAKE}": "@rules_foreign_cc//tools/build_defs:configure.bzl",
                "%{ARGS}": ""
            }
            _tpl(repository_ctx, "libuuid.BUILD", substitutions)
            substitutions["%{TARGETS}"] = ""
            substitutions["%{STATIC_LIB_ARG}"] = "static_libraries"
            _tpl(repository_ctx, "openssl.BUILD", substitutions)
        else:
            substitutions = {
                "%{RULES_FOREIGN_CC_MAKE}": "@rules_foreign_cc//foreign_cc:defs.bzl",
                "%{ARGS}": "args = foreign_make_args(),"
            }
            _tpl(repository_ctx, "libuuid.BUILD", substitutions)
            substitutions["%{TARGETS}"] = "targets = [\"\", \"install_sw\"],"
            substitutions["%{STATIC_LIB_ARG}"] = "out_static_libs"
            _tpl(repository_ctx, "openssl.BUILD", substitutions)
        repository_ctx.template("blade_service_common_workspace.bzl", Label("//bazel/blade_service_common:blade_service_common_workspace.bzl.tpl"), {
        })
    else:
        repository_ctx.template("blade_service_common_workspace.bzl", Label("//bazel/blade_service_common:blade_service_common_empty_workspace.bzl.tpl"), {
        })

    repository_ctx.template("BUILD", Label("//bazel/blade_service_common:BUILD.tpl"), {
    })

blade_service_common_configure = repository_rule(
    implementation = _blade_service_common_impl,
    environ = [
        _IS_PLATFORM_ALIBABA,
        _IS_INTERNAL_SERVING,
    ],
)
