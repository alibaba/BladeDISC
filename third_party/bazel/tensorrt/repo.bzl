_TENSORRT_INSTALL_PATH = "TENSORRT_INSTALL_PATH"

def _impl(repo_ctx):
    tensorrt_path = repo_ctx.os.environ.get(_TENSORRT_INSTALL_PATH, None)
    if tensorrt_path == None:
        fail("Please set the tensorrt library path via env var: {}".format(_TENSORRT_INSTALL_PATH))

    repo_ctx.symlink(Label("//bazel/tensorrt:trt.BUILD.tpl"), "BUILD")
    repo_ctx.symlink(tensorrt_path + "/include", "include")
    repo_ctx.symlink(tensorrt_path + "/lib", "lib")
    #repo_ctx.symlink(tensorrt_path + "/lib64", "lib64")

tensorrt_configure = repository_rule(
    implementation = _impl,
    local = True,
    environ = [_TENSORRT_INSTALL_PATH],
)
