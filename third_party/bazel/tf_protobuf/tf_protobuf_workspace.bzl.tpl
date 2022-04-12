load("@org_tensorflow//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def _tf_protobuf_workspace():
    tf_http_archive(
        name = "tf_protobuf",
        patch_file = ["//third_party/bazel/tf_protobuf:protobuf.patch"],
        sha256 = "%{TF_PROTOBUF_SHA256}",
        strip_prefix = "%{TF_PROTOBUF_STRIP_PREFIX}",
        system_build_file = "//third_party/bazel/tf_protobuf:protobuf.BUILD",
        system_link_files = {
            "//third_party/bazel/tf_protobuf:protobuf.bzl": "protobuf.bzl",
        },
        urls = tf_mirror_urls("%{TF_PROTOBUF_URL}"),
    )

def workspace():
    _tf_protobuf_workspace()

tf_protobuf_workspace = workspace
