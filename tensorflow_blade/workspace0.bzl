# Import repository rules.
load("@org_third_party//bazel/blade_helper:blade_helper_configure.bzl", "blade_helper_configure")
load("@org_third_party//bazel:common.bzl", "maybe_http_archive")
load("@org_third_party//bazel/mkl:mkl_configure.bzl", "mkl_configure")
load("@org_third_party//bazel/tensorrt:repo.bzl", "tensorrt_configure")
load("@org_third_party//bazel/tf:tf_configure.bzl", "tf_configure")

load("@org_tensorflow//third_party/gpus:cuda_configure.bzl", "cuda_configure")

# Import external repository rules.
load("@pybind11_bazel//:python_configure.bzl", "python_configure")
load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")

def _tf_blade_repositories():
    maybe_http_archive(
        name = "bazel_skylib",
        sha256 = "1c531376ac7e5a180e0237938a2536de0c54d93f5c278634818e0efc952dd56c",
        urls = [
            "http://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/build_deps/bazel-skylib-1.0.3.tar.gz",
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/releases/download/1.0.3/bazel-skylib-1.0.3.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.3/bazel-skylib-1.0.3.tar.gz",
        ],
    )

    maybe_http_archive(
        name = "rules_python",
        sha256 = "b6d46438523a3ec0f3cead544190ee13223a52f6a6765a29eae7b7cc24cc83a0",
        urls = [
            "https://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/share/rules_python-0.1.0.tar.gz",
        ],
    )

    maybe_http_archive(
        name = "cub_archive",
        build_file = "@org_third_party//bazel/third_party:cub.BUILD",
        sha256 = "162514b3cc264ac89d91898b58450190b8192e2af1142cf8ccac2d59aa160dda",
        strip_prefix = "cub-1.9.9",
        urls = [
            "http://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/build_deps/cub_archive/1.9.9.zip",
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/NVlabs/cub/archive/1.9.9.zip",
            "https://github.com/NVlabs/cub/archive/1.9.9.zip",
        ],
    )

    rules_foreign_cc_dependencies(
        register_built_tools = False,  # do not build cmake/make from source,
        register_default_tools = False,  # nor download from official site,
        register_preinstalled_tools = True,  # just use the pre-installed.
    )


def _tf_blade_toolchains():
    tf_configure(name = "local_config_tf")

    cuda_configure(name = "local_config_cuda")

    tensorrt_configure(name = "blade_config_tensorrt")

    mkl_configure(name = "local_config_mkl")

    blade_helper_configure(name = "local_config_blade_helper")

    python_configure(name = "local_config_python")


def workspace():
    _tf_blade_repositories()
    _tf_blade_toolchains()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
tf_blade_workspace0 = workspace
