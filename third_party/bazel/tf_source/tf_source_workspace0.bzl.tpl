load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

def _workspaces():
    tf_workspace(path_prefix = "", tf_repo_name = "org_tensorflow")

def workspace():
    _workspaces()

tf_source_workspace0 = workspace
