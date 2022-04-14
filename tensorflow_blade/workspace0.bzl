# Import repository rules.
load("@org_third_party//bazel/blade_disc_helper:blade_disc_helper_configure.bzl", "blade_disc_helper_configure")
load("@org_third_party//bazel:common.bzl", "maybe_http_archive")
load("@org_third_party//bazel/mkl:mkl_configure.bzl", "mkl_configure")
load("@org_third_party//bazel/tensorrt:repo.bzl", "tensorrt_configure")
load("@org_third_party//bazel/tf:tf_configure.bzl", "tf_configure")
load("@org_third_party//bazel/tf_protobuf:tf_protobuf_configure.bzl", "tf_protobuf_configure")
load("@org_third_party//bazel/cuda_supplement:cuda_supplement_configure.bzl", "cuda_supplement_configure")

load("@org_tensorflow//third_party/gpus:cuda_configure.bzl", "cuda_configure")

# Import external repository rules.
load("@pybind11_bazel//:python_configure.bzl", "python_configure")
load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

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

    maybe_http_archive(
        name = "rules_java",
        sha256 = "f5a3e477e579231fca27bf202bb0e8fbe4fc6339d63b38ccb87c2760b533d1c3",
        strip_prefix = "rules_java-981f06c3d2bd10225e85209904090eb7b5fb26bd",
        urls = ["http://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/bazelbuild/rules_java/archive/981f06c3d2bd10225e85209904090eb7b5fb26bd.tar.gz"],
    )

    # mkldnn cmake external rules
    maybe_http_archive(
        name = "mkl_static",
        build_file = "@local_config_mkldnn//:mkl_static.BUILD",
        sha256 = "b0f4f03c5a2090bc1194f348746396183cfb63a5a379d6e86f7fa89006abe28b",
        urls = [
            "https://hlomodule.oss-cn-zhangjiakou.aliyuncs.com/mkl_package/mkl-static-2022.0.1-intel_117.tar.bz2",
            "https://hlomodule.oss-cn-zhangjiakou.aliyuncs.com/mkl_package/mkl-static-2022.0.1-intel_117.tar.bz2",
        ],
    )

    maybe_http_archive(
        name = "mkl_include",
        build_file = "@local_config_mkldnn//:mkl_include.BUILD",
        sha256 = "3df729b9fa66f2e1e566c70baa6799b15c9d0e5d3890b9bd084e02299af25002",
        urls = [
            "https://hlomodule.oss-cn-zhangjiakou.aliyuncs.com/mkl_package/mkl-include-2022.0.1-h8d4b97c_803.tar.bz2",
            "https://hlomodule.oss-cn-zhangjiakou.aliyuncs.com/mkl_package/mkl-include-2022.0.1-h8d4b97c_803.tar.bz2",
        ],
    )

    native.new_local_repository(
        name = "onednn",
        build_file = "@local_config_mkldnn//:onednn.BUILD",
        path = "../tao/third_party/mkldnn"
    )

    # for aarch64 related acl library
    new_git_repository(
        name = "acl_compute_library",
        build_file = "@local_config_mkldnn//:compute_library.BUILD",
        remote = "https://github.com/ARM-software/ComputeLibrary.git",
        tag = "v22.02",
        init_submodules = True,
    )

def _tf_blade_toolchains():
    tf_configure(name = "local_config_tf")

    cuda_supplement_configure(name = "local_config_cuda_supplement")

    tensorrt_configure(name = "local_config_trt")

    mkl_configure(name = "local_config_mkl")

    blade_disc_helper_configure(name = "local_config_blade_disc_helper")

    python_configure(name = "local_config_python")

    tf_protobuf_configure(name = "local_config_tf_protobuf")

def workspace():
    _tf_blade_repositories()
    _tf_blade_toolchains()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
tf_blade_workspace0 = workspace
