load("@bazel_skylib//rules:common_settings.bzl", "string_flag", "string_list_flag")

package(default_visibility = ["//visibility:public"])

string_flag(
    name = "device",
    build_setting_default = "gpu",
    values = [
        "gpu",
        "cpu",
    ],
)

string_list_flag(
    name = "framework",
    build_setting_default = [],
)

config_setting(
    name = "gpu",
    flag_values = {":device": "gpu"},
)

config_setting(
    name = "cpu",
    flag_values = {":device": "cpu"},
)
