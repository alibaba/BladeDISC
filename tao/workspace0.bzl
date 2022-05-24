# Import repository rules.
load("@org_third_party//bazel/blade_disc_helper:blade_disc_helper_configure.bzl", "blade_disc_helper_configure")
load("@org_third_party//bazel:common.bzl", "maybe_http_archive")
load("@org_third_party//bazel/tf:tf_configure.bzl", "tf_configure")
load("@org_third_party//bazel/tf_source:tf_source_configure.bzl", "tf_source_configure")
load("@org_third_party//bazel/tf_protobuf:tf_protobuf_configure.bzl", "tf_protobuf_configure")
load("@org_third_party//bazel/blade_service_common:blade_service_common_configure.bzl", "blade_service_common_configure")

# Import external repository rules.
load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def _tao_bridge_repositories():
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
        urls = ["http://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/bazelbuild/rules_java/archive/981f06c3d2bd10225e85209904090eb7b5fb26bd.tar.gz"],
    )

    native.new_local_repository(
        name = "blade_gemm",
        build_file = "@org_third_party//bazel/third_party:blade_gemm.BUILD",
        path = "../../platform_alibaba/blade_gemm"
    )

def _tao_bridge_toolchains():
    tf_configure(name = "local_config_tf")

    tf_protobuf_configure(name = "local_config_tf_protobuf")

    blade_service_common_configure(name = "local_config_blade_service_common")

    tf_source_configure(name = "local_config_tf_source")

    blade_disc_helper_configure(name = "local_config_blade_disc_helper")

def workspace():
    _tao_bridge_repositories()
    _tao_bridge_toolchains()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
tao_bridge_workspace0 = workspace
