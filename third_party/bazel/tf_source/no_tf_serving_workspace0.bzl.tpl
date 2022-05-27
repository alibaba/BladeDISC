load("@local_config_tf_protobuf//:tf_protobuf_workspace.bzl", "tf_protobuf_workspace")

def _workspace():
    tf_protobuf_workspace()

def workspace():
    _workspace()

tf_source_workspace0 = workspace
