load("@org_third_party//bazel:common.bzl", "tf_serving_http_archive")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _dependencies():
    # START: Upstream TensorFlow dependencies
    # TensorFlow build depends on these dependencies.
    # Needs to be in-sync with TensorFlow sources.
    http_archive(
        name = "io_bazel_rules_closure",
        sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
        strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
            "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
        ],
    )

    http_archive(
        name = "bazel_skylib",
        sha256 = "1dde365491125a3db70731e25658dfdd3bc5dbdfd11b840b3e987ecf043c7ca0",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
        ],
    )  # https://github.com/bazelbuild/bazel-skylib/releases

def _repositories():
    tf_serving_http_archive(
        name = "org_tensorflow",
        patch = "@local_config_tf_source//:tf_source_code.patch",
        sha256 = "%{TF_SOURCE_SHA256}",
        git_commit = "%{TF_SOURCE_GIT_COMMIT}",
    )

def workspace():
    _repositories()
    _dependencies()

tf_source_workspace1 = workspace
