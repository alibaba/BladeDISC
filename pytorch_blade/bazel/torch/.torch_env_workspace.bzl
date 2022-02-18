
def torch_env_workspace():
    native.new_local_repository(
        name = "local_org_torch",
        path = "/usr/local/lib/python3.6/site-packages/torch",
        build_file = "@//bazel/torch:BUILD",
    )

def torch_pybind11_dir():
    return "/usr/local/lib/python3.6/site-packages/torch/include/pybind11"
