package(default_visibility = ["//visibility:public"])

config_setting(
    name = "is_platform_alibaba",
    define_values = {"is_platform_alibaba": "true"},
)

config_setting(
    name = "is_mkldnn",
    define_values = {"is_mkldnn": "true"},
)

config_setting(
    name = "disc_aarch64",
    define_values = {"disc_aarch64": "true"},
)

config_setting(
    name = "disc_x86",
    define_values = {"disc_x86": "true"},
)

config_setting(
    name = "is_cxx11_abi",
    define_values = {"is_cxx11_abi": "true"},
)

config_setting(
    name = "is_internal_serving",
    define_values = {"is_internal_serving": "true"},
)
