# Import repository rules.
load("@org_third_party//bazel/blade_disc_helper:blade_disc_helper_configure.bzl", "blade_disc_helper_configure")
load("@org_third_party//bazel:common.bzl", "maybe_http_archive")
load("@org_third_party//bazel/tf:tf_configure.bzl", "tf_configure")
load("@org_third_party//bazel/tf_protobuf:tf_protobuf_configure.bzl", "tf_protobuf_configure")
load("@org_third_party//bazel/blade_service_common:blade_service_common_configure.bzl", "blade_service_common_configure")

load("@org_tensorflow//third_party/gpus:cuda_configure.bzl", "cuda_configure")
load("@org_tensorflow//third_party/gpus:rocm_configure.bzl", "rocm_configure")

# Import external repository rules.
load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def _tao_bridge_repositories():
    maybe_http_archive(
        name = "bazel_skylib",
        sha256 = "1c531376ac7e5a180e0237938a2536de0c54d93f5c278634818e0efc952dd56c",
        urls = [
            "http://pai-blade.oss-accelerate.aliyuncs.com/build_deps/bazel-skylib-1.0.3.tar.gz",
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/releases/download/1.0.3/bazel-skylib-1.0.3.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.3/bazel-skylib-1.0.3.tar.gz",
        ],
    )

    maybe_http_archive(
        name = "cub_archive",
        build_file = "@org_third_party//bazel/third_party:cub.BUILD",
        sha256 = "162514b3cc264ac89d91898b58450190b8192e2af1142cf8ccac2d59aa160dda",
        strip_prefix = "cub-1.9.9",
        urls = [
            "http://pai-blade.oss-accelerate.aliyuncs.com/build_deps/cub_archive/1.9.9.zip",
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/NVlabs/cub/archive/1.9.9.zip",
            "https://github.com/NVlabs/cub/archive/1.9.9.zip",
        ],
    )

    rules_foreign_cc_dependencies(
        register_built_tools = False,  # do not build cmake/make from source,
        register_default_tools = False,  # nor download from official site,
        register_preinstalled_tools = True,  # just use the pre-installed.
    )

    # work around for rules_jave download failures
    maybe_http_archive(
        name = "rules_java",
        sha256 = "f5a3e477e579231fca27bf202bb0e8fbe4fc6339d63b38ccb87c2760b533d1c3",
        strip_prefix = "rules_java-981f06c3d2bd10225e85209904090eb7b5fb26bd",
        urls = ["http://pai-blade.oss-accelerate.aliyuncs.com/bazelbuild/rules_java/archive/981f06c3d2bd10225e85209904090eb7b5fb26bd.tar.gz"],
    )

    native.new_local_repository(
        name = "onednn",
        build_file = "@local_config_onednn//:onednn.BUILD",
        path = "third_party/mkldnn"
    )

    native.new_local_repository(
        name = "blade_gemm",
        build_file = "@org_third_party//bazel/third_party:blade_gemm.BUILD",
        path = "../../platform_alibaba/blade_gemm"
    )

def _tao_bridge_toolchains():
    tf_configure(name = "local_config_tf")

    tf_protobuf_configure(name = "local_config_tf_protobuf")

    blade_disc_helper_configure(name = "local_config_blade_disc_helper")

    blade_service_common_configure(name = "local_config_blade_service_common")

    cuda_configure(name = "local_config_cuda")

    rocm_configure(name = "local_config_rocm")

def workspace():
    _tao_bridge_repositories()
    _tao_bridge_toolchains()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
tao_bridge_workspace0 = workspace
