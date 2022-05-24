def _tao_bridge_repositories():
    native.local_repository(
        name = "org_third_party",
        path = "../third_party/",
    )
    native.local_repository(
        name = "org_tao_compiler",
        path = "../tao_compiler/",
    )

def workspace():
    _tao_bridge_repositories()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
tao_bridge_workspace2 = workspace
