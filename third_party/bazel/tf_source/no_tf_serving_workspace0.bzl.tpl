load("@local_config_tf_protobuf//:tf_protobuf_workspace.bzl", "tf_protobuf_workspace")
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")
load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")
load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")
load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

def _workspaces():
    tf_protobuf_workspace()

    # Initialize TensorFlow's external dependencies when build with tf wheels.
    tf_workspace3()

    tf_workspace2()

    tf_workspace1()

    tf_workspace0()

def workspace():
    _workspaces()

tf_source_workspace0 = workspace
