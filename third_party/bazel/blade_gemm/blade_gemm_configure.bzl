load("//bazel:common.bzl", "get_host_environ")

_IS_PLATFORM_ALIBABA = "IS_PLATFORM_ALIBABA"
_BLADE_GEMM_REPO_PATH = "BLADE_GEMM_REPO_PATH"
_CUTLASS_NVCC_ARCHS = "CUTLASS_NVCC_ARCHS"
_CUTLASS_LIBRARY_KERNELS = "CUTLASS_LIBRARY_KERNELS"
_CUTLASS_CUDACXX = "CUTLASS_CUDACXX"

def _blade_gemm_configure_impl(repository_ctx):
    is_platform_alibaba = repository_ctx.os.environ[_IS_PLATFORM_ALIBABA].lower() if _IS_PLATFORM_ALIBABA in repository_ctx.os.environ else "0"
    if is_platform_alibaba in ["1", "true", "on"]:
        repository_ctx.template(
            "blade_gemm_workspace.bzl",
            Label("//bazel/blade_gemm:blade_gemm_workspace.bzl.tpl"),
            {
                "%{BLADE_GEMM_REPO_PATH}": get_host_environ(repository_ctx, _BLADE_GEMM_REPO_PATH, "../../platform_alibaba/blade_gemm/"),
            },
        )

        repository_ctx.template("blade_gemm.BUILD", Label("//bazel/blade_gemm:blade_gemm.BUILD.tpl"),
            {
                "%{CUTLASS_NVCC_ARCHS}": get_host_environ(repository_ctx, _CUTLASS_NVCC_ARCHS, "80"),
                "%{CUTLASS_LIBRARY_KERNELS}": get_host_environ(repository_ctx, _CUTLASS_LIBRARY_KERNELS, "s1688tf32gemm,f16_s1688gemm_f16,f16_s16816gemm_f16,s16816tf32gemm"),
                "%{CUTLASS_CUDACXX}": get_host_environ(repository_ctx, _CUTLASS_CUDACXX, "/usr/local/cuda-11.6/bin/nvcc"),
            },
        )
        repository_ctx.template("BUILD", Label("//bazel/blade_gemm:BUILD.tpl"), {})
    else:
        repository_ctx.template("blade_gemm_workspace.bzl", Label("//bazel/blade_gemm:blade_gemm_empty_workspace.bzl.tpl"), {})
        repository_ctx.template("BUILD", Label("//bazel/blade_gemm:BUILD.tpl"), {})

blade_gemm_configure = repository_rule(
    implementation = _blade_gemm_configure_impl,
    environ = [
        _IS_PLATFORM_ALIBABA,
        _BLADE_GEMM_REPO_PATH,
        _CUTLASS_NVCC_ARCHS,
        _CUTLASS_LIBRARY_KERNELS,
        _CUTLASS_CUDACXX,
    ],
)
