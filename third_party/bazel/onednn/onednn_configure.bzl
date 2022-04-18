load("//bazel:common.bzl", "get_env_bool_value", "get_host_environ")

_BUILD_WITH_AARCH64 = "BUILD_WITH_AARCH64"
_BUILD_WITH_MKLDNN = "BUILD_WITH_MKLDNN"
_IF_CXX11_ABI = "IF_CXX11_ABI"

def _onednn_impl(repository_ctx):
    with_mkldnn = get_env_bool_value(repository_ctx, _BUILD_WITH_MKLDNN)
    if with_mkldnn:
        if_cxx11_abi = get_env_bool_value(repository_ctx, _IF_CXX11_ABI)
        with_aarch64 = get_env_bool_value(repository_ctx, _BUILD_WITH_AARCH64)
        if with_aarch64:
            acl_root = repository_ctx.path(Label("@acl_compute_library//:Makefile")).dirname
        repository_ctx.template("onednn.BUILD", Label("//bazel/onednn:onednn.BUILD.tpl"), {
            "%{ACL_ROOT}": "\"ACL_ROOT_DIR\": \"{}\",".format(acl_root) if with_aarch64 else "",
            "%{ACL_SETTING}": "\"DNNL_AARCH64_USE_ACL\": \"ON\"," if with_aarch64 else "",
            "%{CXX11_SETTING}": "\"USE_CXX11_ABI\": \"ON\"" if if_cxx11_abi else "",
        })
        repository_ctx.template("BUILD", Label("//bazel/onednn:BUILD.tpl"), {})

onednn_configure = repository_rule(
    implementation = _onednn_impl,
    environ = [
        _BUILD_WITH_AARCH64,
        _BUILD_WITH_MKLDNN,
        _IF_CXX11_ABI,
    ],
)
