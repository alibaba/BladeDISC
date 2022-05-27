# Import repository rules.
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@org_third_party//bazel/tf_source:tf_source_configure.bzl", "tf_source_configure")
load("@org_third_party//bazel/tf_protobuf:tf_protobuf_configure.bzl", "tf_protobuf_configure")

def _tao_bridge_repositories():
    http_archive(
        name = "rules_foreign_cc",
        sha256 = "33a5690733c5cc2ede39cb62ebf89e751f2448e27f20c8b2fbbc7d136b166804",
        strip_prefix = "rules_foreign_cc-0.5.1",
        urls = [
            "http://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/build_deps/rules_foreign_cc/0.5.1.tar.gz",
            "https://github.com/bazelbuild/rules_foreign_cc/archive/0.5.1.tar.gz",
        ],
    )

    tf_protobuf_configure(name = "local_config_tf_protobuf")

    tf_source_configure(name = "local_config_tf_source")


def workspace():
    _tao_bridge_repositories()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
tao_bridge_workspace1 = workspace
