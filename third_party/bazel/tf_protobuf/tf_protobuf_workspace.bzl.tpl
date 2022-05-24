load("@org_third_party//bazel:common.bzl", "tf_mirror_urls", "tf_http_archive")

def _tf_protobuf_workspace():
    # NOTE(lanbo.llb): tf_http_archive will check if rule with same already exists,
    # if so, the duplicate rule by `tf_http_archive` will not be used.
    # Our own version of protobuf will be used instead of tf_community's.
    # Make sure this rule is loaded first when building tf's custom op/passes.
    tf_http_archive(
        name = "com_google_protobuf",
        patch_file = ["@local_config_tf_protobuf//:protobuf.patch"],
        sha256 = "%{TF_PROTOBUF_SHA256}",
        strip_prefix = "%{TF_PROTOBUF_STRIP_PREFIX}",
        system_build_file = "@local_config_tf_protobuf//:protobuf.BUILD",
        system_link_files = {
            "@local_config_tf_protobuf//:protobuf.bzl": "protobuf.bzl",
        },
        urls = tf_mirror_urls("%{TF_PROTOBUF_URL}"),
    )

def workspace():
    _tf_protobuf_workspace()

tf_protobuf_workspace = workspace
