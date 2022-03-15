load("@bazel_skylib//rules:common_settings.bzl", "string_flag", "string_list_flag")

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

config_setting(
    name = "tf_supported",
    flag_values = {":framework": "tf"},
)

py_library(
    name = "context",
    srcs = [],
    imports = ["."],
    visibility = ["//visibility:public"],
)
