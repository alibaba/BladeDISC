load("//bazel:common.bzl", "files_exist")

_TF_CUDA_HOME = "TF_CUDA_HOME"

def _create_dummy_repository(repo_ctx):
    repo_ctx.symlink(Label("//bazel/cuda_static:dummy.BUILD.tpl"), "BUILD")
    repo_ctx.template("build_defs.bzl", Label("//bazel/cuda_static:build_defs.bzl.tpl"), {
        "%{IF_HAS_CUBLASLT}": "False",
    })

def _impl(repo_ctx):
    cuda_path = repo_ctx.os.environ.get(_TF_CUDA_HOME, None)
    if cuda_path != None:
        repo_ctx.symlink(Label("//bazel/cuda_static:cuda_static.BUILD.tpl"), "BUILD")
        repo_ctx.symlink(cuda_path + "/lib64", "lib64")
        repo_ctx.template("build_defs.bzl", Label("//bazel/cuda_static:build_defs.bzl.tpl"), {
            "%{IF_HAS_CUBLASLT}": "True" if files_exist(repo_ctx, ["lib64/libcublasLt_static.a"])[0] else "False",
        })
    else:
        _create_dummy_repository(repo_ctx)

cuda_static_configure = repository_rule(
    implementation = _impl,
    local = True,
    environ = [_TF_CUDA_HOME],
)
