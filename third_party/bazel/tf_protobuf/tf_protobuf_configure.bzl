"""Setup TensorFlow as external dependency"""
load("//bazel:common.bzl", "get_env_bool_value")

_TF_PROTOBUF_VERSION = "TF_PROTOBUF_VERSION"

def _tf_protobuf_configure_impl(repository_ctx):
    protobuf_version = repository_ctx.os.environ[_TF_PROTOBUF_VERSION]
    protobuf_dict = {
        "3.9.2": [  # tf-2.4.0/tf-2.8.0 both 3.9.2
            "cfcba2df10feec52a84208693937c17a4b5df7775e1635c1e3baffc487b24c9b",  # sha256
            "protobuf-3.9.2",  # strip_prefix
            "https://github.com/protocolbuffers/protobuf/archive/v3.9.2.zip",  # url
        ],
        "3.8.0": [  # tf-1.15-rc3
            "b9e92f9af8819bbbc514e2902aec860415b70209f31dfc8c4fa72515a5df9d59",  #sha256
            "protobuf-310ba5ee72661c081129eb878c1bbcec936b20f0",
            "https://github.com/protocolbuffers/protobuf/archive/310ba5ee72661c081129eb878c1bbcec936b20f0.tar.gz"
        ]
    }
    protobuf_info = protobuf_dict.get(protobuf_version)
    if protobuf_info == None:
        fail("The protobuf version - {}, acquired from tensorflow wheel is not supported now!".format(protobuf_version))
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
    repository_ctx.template("protobuf.BUILD", Label("//bazel/tf_protobuf:protobuf-{}.BUILD.tpl".format(protobuf_version)), {})
    repository_ctx.template("protobuf.bzl", Label("//bazel/tf_protobuf:protobuf-{}.bzl.tpl".format(protobuf_version)), {})
    repository_ctx.template("protobuf.patch", Label("//bazel/tf_protobuf:protobuf-{}.patch.tpl".format(protobuf_version)), {})

tf_protobuf_configure = repository_rule(
    implementation = _tf_protobuf_configure_impl,
    environ = [
        _TF_PROTOBUF_VERSION,
    ],
)
