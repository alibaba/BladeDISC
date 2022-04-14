load("//bazel:common.bzl", "get_env_bool_value", "get_host_environ")

_ACL_ROOT_PATH = "ACL_ROOT_PATH"
_BUILD_WITH_AARCH64 = "BUILD_WITH_AARCH64"
_BUILD_WITH_MKLDNN = "BUILD_WITH_MKLDNN"
_IF_CXX11_ABI = "IF_CXX11_ABI"

def _mkldnn_impl(repository_ctx):
    with_mkldnn = get_env_bool_value(repository_ctx, _BUILD_WITH_MKLDNN)
    if with_mkldnn:
        if_cxx11_abi = get_env_bool_value(repository_ctx, _IF_CXX11_ABI)
        with_aarch64 = get_env_bool_value(repository_ctx, _BUILD_WITH_AARCH64)
        if with_aarch64:
            acl_root = get_host_environ(repository_ctx, _ACL_ROOT_PATH)
        repository_ctx.template("onednn.BUILD", Label("//bazel/mkldnn:onednn.BUILD.tpl"), {
            "%{ACL_SETTING}": "\"DNNL_AARCH64_USE_ACL\": \"ON\"," if with_aarch64 else "",
            "%{ACL_ROOT}": "\"ACL_ROOT_DIR\": \"{}\",".format(acl_root) if with_aarch64 else "",
            "%{CXX11_SETTING}": "\"USE_CXX11_ABI\": \"ON\"" if if_cxx11_abi else "",
        })
        repository_ctx.template("mkl_include.BUILD", Label("//bazel/mkldnn:mkl_include.BUILD.tpl"), {})
        repository_ctx.template("mkl_static.BUILD", Label("//bazel/mkldnn:mkl_static.BUILD.tpl"), {})
        repository_ctx.template("compute_library.BUILD", Label("//bazel/mkldnn:compute_library.BUILD.tpl"), {})
        repository_ctx.template("compute_library.sh", Label("//bazel/mkldnn:compute_library.sh.tpl"), {})
        repository_ctx.template("BUILD", Label("//bazel/mkldnn:BUILD.tpl"), {})

mkldnn_configure = repository_rule(
    implementation = _mkldnn_impl,
    environ = [
        _ACL_ROOT_PATH,
        _BUILD_WITH_AARCH64,
        _BUILD_WITH_MKLDNN,
        _IF_CXX11_ABI,
    ],
)
