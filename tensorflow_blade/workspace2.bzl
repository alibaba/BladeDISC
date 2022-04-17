def _tf_blade_repositories():
    native.local_repository(
        name = "org_tensorflow",
        path = "../tf_community/",
    )
    native.local_repository(
        name = "org_third_party",
        path = "../third_party/",
    )
    native.local_repository(
        name = "org_tao_compiler",
        path = "../tao_compiler/",
    )
    native.local_repository(
        name = "org_tao_bridge",
        path = "../tao/",
    )


def workspace():
    _tf_blade_repositories()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
tf_blade_workspace2 = workspace
