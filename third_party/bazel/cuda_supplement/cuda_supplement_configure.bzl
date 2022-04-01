load("//bazel:common.bzl", "files_exist")

_TF_CUDA_HOME = "TF_CUDA_HOME"

def _create_dummy_repository(repo_ctx):
    repo_ctx.symlink(Label("//bazel/cuda_supplement:dummy.BUILD.tpl"), "BUILD")
    repo_ctx.template("build_defs.bzl", Label("//bazel/cuda_supplement:build_defs.bzl.tpl"), {
        "%{IF_HAS_CUBLASLT}": "False",
    })

def _cc_import_cublaslt():
    return """
cc_import(
    name = "cublasLt_static",
    static_library = "lib64/libcublasLt_static.a",
)"""

def _impl(repo_ctx):
    cuda_path = repo_ctx.os.environ.get(_TF_CUDA_HOME, None)
    if cuda_path != None:
        if_has_cublaslt = files_exist(repo_ctx, ["lib64/libcublasLt_static.a"])[0]
        repo_ctx.template("BUILD", Label("//bazel/cuda_supplement:cuda_supplement.BUILD.tpl"), {
            "%{import_cublaslt}": _cc_import_cublaslt() if if_has_cublaslt else "",
        })
        repo_ctx.symlink(cuda_path + "/lib64", "lib64")
        repo_ctx.template("build_defs.bzl", Label("//bazel/cuda_supplement:build_defs.bzl.tpl"), {
            "%{IF_HAS_CUBLASLT}": "True" if if_has_cublaslt else "False",
        })
    else:
        _create_dummy_repository(repo_ctx)

cuda_supplement_configure = repository_rule(
    implementation = _impl,
    local = True,
    environ = [_TF_CUDA_HOME],
)
