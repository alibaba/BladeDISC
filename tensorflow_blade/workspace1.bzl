# Import repository rules.
load("@org_third_party//bazel:common.bzl", "maybe_http_archive")

def _tf_blade_repositories():
    maybe_http_archive(
        name = "rules_foreign_cc",
        sha256 = "33a5690733c5cc2ede39cb62ebf89e751f2448e27f20c8b2fbbc7d136b166804",
        strip_prefix = "rules_foreign_cc-0.5.1",
        urls = [
            "http://pai-blade.oss-accelerate.aliyuncs.com/build_deps/rules_foreign_cc/0.5.1.tar.gz",
            "https://github.com/bazelbuild/rules_foreign_cc/archive/0.5.1.tar.gz",
        ],
    )

    maybe_http_archive(
        name = "pybind11_bazel",
        sha256 = "a5666d950c3344a8b0d3892a88dc6b55c8e0c78764f9294e806d69213c03f19d",
        strip_prefix = "pybind11_bazel-26973c0ff320cb4b39e45bc3e4297b82bc3a6c09",
        urls = [
            "http://pai-blade.oss-accelerate.aliyuncs.com/build_deps/pybind11_bazel/26973c0ff320cb4b39e45bc3e4297b82bc3a6c09.zip",
            "https://github.com/pybind/pybind11_bazel/archive/26973c0ff320cb4b39e45bc3e4297b82bc3a6c09.zip",
        ],
    )

    maybe_http_archive(
        name = "pybind11",
        build_file = "@pybind11_bazel//:pybind11.BUILD",
        sha256 = "3a3b7b651afab1c5ba557f4c37d785a522b8030dfc765da26adc2ecd1de940ea",
        strip_prefix = "pybind11-2.2.3",
        urls = [
            "http://pai-blade.oss-accelerate.aliyuncs.com/build_deps/pybind11/v2.2.3.tar.gz",
            "https://github.com/pybind/pybind11/archive/refs/tags/v2.2.3.tar.gz",
        ],
    )

def workspace():
    _tf_blade_repositories()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
tf_blade_workspace1 = workspace
