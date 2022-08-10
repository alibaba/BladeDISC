# Import repository rules.
load("@org_third_party//bazel:common.bzl", "maybe_http_archive")

def _tao_bridge_repositories():
    maybe_http_archive(
        name = "rules_foreign_cc",
        sha256 = "33a5690733c5cc2ede39cb62ebf89e751f2448e27f20c8b2fbbc7d136b166804",
        strip_prefix = "rules_foreign_cc-0.5.1",
        urls = [
            "http://pai-blade.oss-accelerate.aliyuncs.com/build_deps/rules_foreign_cc/0.5.1.tar.gz",
            "https://github.com/bazelbuild/rules_foreign_cc/archive/0.5.1.tar.gz",
        ],
    )

def workspace():
    _tao_bridge_repositories()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
tao_bridge_workspace1 = workspace
