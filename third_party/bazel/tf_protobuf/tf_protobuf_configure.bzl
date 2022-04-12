"""Setup TensorFlow as external dependency"""
load("//bazel:common.bzl", "get_env_bool_value")

_TF_PROTOBUF_VERSION = "TF_PROTOBUF_VERSION"

def _tf_protobuf_configure_impl(repository_ctx):
    protobuf_version = repository_ctx.os.environ[_TF_PROTOBUF_VERSION]
    # for tensorflow==1.15.0 and tensorflow==2.4.0, the protobuf versions are the same, 3.9.2
    protobuf_dict = {
        "3.9.2": [
            "cfcba2df10feec52a84208693937c17a4b5df7775e1635c1e3baffc487b24c9b",  # sha256
            "protobuf-3.9.2",  # strip_prefix
            "https://github.com/protocolbuffers/protobuf/archive/v3.9.2.zip",  # url
        ],
    }
    protobuf_info = protobuf_dict.get(protobuf_version)
    if protobuf_info == None:
        fail("The protobuf acquired from tensorflow wheel is not supported now!")
    repository_ctx.template(
        "tf_protobuf_workspace.bzl",
        Label("//bazel/tf_protobuf:tf_protobuf_workspace.bzl.tpl"),
        {
            "%{TF_PROTOBUF_SHA256}": protobuf_info[0],
            "%{TF_PROTOBUF_STRIP_PREFIX}": protobuf_info[1],
            "%{TF_PROTOBUF_URL}": protobuf_info[2],
        },
    )

    repository_ctx.template("BUILD", Label("//bazel/tf_protobuf:BUILD.tpl"), {})
    repository_ctx.template("protobuf.BUILD", Label("//bazel/tf_protobuf:protobuf.BUILD.tpl"), {})
    repository_ctx.template("protobuf.bzl", Label("//bazel/tf_protobuf:protobuf.bzl.tpl"), {})
    repository_ctx.template("protobuf.patch", Label("//bazel/tf_protobuf:protobuf.patch.tpl"), {})

tf_protobuf_configure = repository_rule(
    implementation = _tf_protobuf_configure_impl,
    environ = [
        _TF_PROTOBUF_VERSION,
    ],
)
