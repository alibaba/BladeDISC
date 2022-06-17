load("@org_third_party//bazel/blade_service_common:blade_service_common_configure.bzl", "blade_service_common_configure")

def workspace_platform_alibaba():
    blade_service_common_configure(name = "local_config_blade_service_common")
