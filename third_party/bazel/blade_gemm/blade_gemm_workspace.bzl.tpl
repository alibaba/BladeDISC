def _blade_gemm_workspace():
    native.new_local_repository(
        name = "blade_gemm",
        build_file = "@local_config_blade_gemm//:blade_gemm.BUILD",
        path = "%{BLADE_GEMM_REPO_PATH}",
    )

def workspace():
    _blade_gemm_workspace()

blade_gemm_workspace = workspace
