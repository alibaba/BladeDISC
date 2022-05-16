load("//bazel:common.bzl", "files_exist")

_TF_CUDA_HOME = "TF_CUDA_HOME"

def _create_dummy_repository(repo_ctx):
    repo_ctx.symlink(Label("//bazel/cuda_supplement:dummy.BUILD.tpl"), "BUILD")
    repo_ctx.template("build_defs.bzl", Label("//bazel/cuda_supplement:build_defs.bzl.tpl"), {
        "%{IF_HAS_CUBLASLT}": "False",
        "%{IF_HAS_CUDNN_STATIC}": "False",
    })

def _impl(repo_ctx):
    cuda_path = repo_ctx.os.environ.get(_TF_CUDA_HOME, None)
    if cuda_path != None:
        if_has_cublaslt, if_has_cudnn_static = files_exist(repo_ctx, [
            cuda_path + "/lib64/libcublasLt_static.a",
            cuda_path + "/lib64/libcudnn_static.a",
        ])
        repo_ctx.template("BUILD", Label("//bazel/cuda_supplement:cuda_supplement.BUILD.tpl"), {})
        repo_ctx.symlink(cuda_path + "/lib64", "lib64")
        repo_ctx.template("build_defs.bzl", Label("//bazel/cuda_supplement:build_defs.bzl.tpl"), {
            "%{IF_HAS_CUBLASLT}": "True" if if_has_cublaslt else "False",
            "%{IF_HAS_CUDNN_STATIC}": "True" if if_has_cudnn_static else "False",
        })
    else:
        _create_dummy_repository(repo_ctx)

cuda_supplement_configure = repository_rule(
    implementation = _impl,
    local = True,
    environ = [_TF_CUDA_HOME],
)
