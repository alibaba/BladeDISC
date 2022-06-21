def _repositories():
    native.local_repository(
        name = "org_tensorflow",
        path = "../tf_community/",
    )

def workspace():
    _repositories()

tf_source_workspace1 = workspace
