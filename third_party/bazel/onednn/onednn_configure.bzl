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
            # getting acl_root is a little bit hacky, since this template action happens when loading WORKSPACE content.
            # At this time, all the build output from @acl_compute_library are not ready yet, trying to get location of
            # @acl_compute_library//:build/libarm_compute.a will not work since libarm_compute.a is not exist.
            # This Makefile exist after @acl_compute_library's download and patch is complete, thus the `Label` like this.
            acl_root = repository_ctx.path(Label("@acl_compute_library//:Makefile")).dirname
        repository_ctx.template("onednn.BUILD", Label("//bazel/onednn:onednn.BUILD.tpl"), {
            "%{ACL_ROOT}": "\"ACL_ROOT_DIR\": \"{}\",".format(acl_root) if with_aarch64 else "",
            "%{ACL_SETTING}": "\"DNNL_AARCH64_USE_ACL\": \"ON\"," if with_aarch64 else "",
            "%{CXX11_SETTING}": "\"USE_CXX11_ABI\": \"ON\"," if if_cxx11_abi else "",
        })
        repository_ctx.template("BUILD", Label("//bazel/onednn:BUILD.tpl"), {})
        repository_ctx.template("onednn.bzl", Label("//bazel/onednn:onednn.bzl.tpl"), {})

onednn_configure = repository_rule(
    implementation = _onednn_impl,
    environ = [
        _BUILD_WITH_AARCH64,
        _BUILD_WITH_MKLDNN,
        _IF_CXX11_ABI,
    ],
)
