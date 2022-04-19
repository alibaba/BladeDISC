_TORCH_INSTALL_PATH = "TORCH_BLADE_TORCH_INSTALL_PATH"
_TORCH_SUBMODULE = "/workspace/torch_disc/pytorch"

def _impl(repo_ctx):
    torch_path = repo_ctx.os.environ.get(_TORCH_INSTALL_PATH, None)
    if torch_path == None:
        fail("Please set the torch library path via env var: {}".format(_TORCH_INSTALL_PATH))

    repo_ctx.symlink(Label("//bazel/torch:torch.BUILD.tpl"), "BUILD")
    repo_ctx.symlink(torch_path + "/include", "include")
    repo_ctx.symlink(torch_path + "/lib", "lib")
    repo_ctx.symlink(_TORCH_SUBMODULE, "pytorch")
    repo_ctx.symlink(_TORCH_SUBMODULE + "/lazy_tensor_core", "lazy_tensor_core")
    repo_ctx.symlink(_TORCH_SUBMODULE + "/torch/csrc/lazy", "ts_include/torch/csrc/lazy")
    repo_ctx.symlink(_TORCH_SUBMODULE + "/torch/csrc/generic", "ts_include/torch/csrc/generic")
    repo_ctx.symlink(_TORCH_SUBMODULE + "/lazy_tensor_core/third_party/computation_client", "ts_include/lazy_tensors/computation_client")

torch_configure = repository_rule(
    implementation = _impl,
    local = True,
    environ = [_TORCH_INSTALL_PATH],
)

def _symlink_files(repo_ctx, fpaths):
    for f in fpaths:
        fpath = repo_ctx.path(f)
        fname = fpath.basename
        repo_ctx.symlink(fpath, fname)

def _impl_pybind11(repo_ctx):
    torch_path = repo_ctx.os.environ.get(_TORCH_INSTALL_PATH, None)
    if torch_path == None:
        fail("Please set the torch library path via env var: {}".format(_TORCH_INSTALL_PATH))

    pybind11_files = repo_ctx.path(torch_path + "/include/pybind11").readdir()
    _symlink_files(repo_ctx, pybind11_files)
    repo_ctx.symlink(repo_ctx.attr.build_file, "BUILD")

torch_pybind11_configure = repository_rule(
    implementation = _impl_pybind11,
    local = True,
    attrs = {"build_file": attr.label(allow_single_file = True)},
    environ = [_TORCH_INSTALL_PATH],
)
