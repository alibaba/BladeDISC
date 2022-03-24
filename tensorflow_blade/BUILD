load("@bazel_skylib//rules:common_settings.bzl", "string_flag", "string_list_flag")
load("@local_config_blade_helper//:build_defs.bzl", "python_bin_path", "if_tensorrt_disabled")
load("//:build_defs.bzl", "device_name")

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

config_setting(
    name = "tf_supported",
    flag_values = {":framework": "tf"},
)

py_library(
    name = "context",
    srcs = [],
    imports = ["."],
)

py_library(
    name = "setup",
    srcs = [
        "setup.py",
    ],
)

py_library(
    name = "version",
    srcs = ["version.py"],
    visibility = ["//visibility:public"],
)

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    args = [
        "--py_bin",
    ] + [
        python_bin_path(),
    ] + [
        "--device",
    ] + device_name() + if_tensorrt_disabled([
        "--skip_trt",
    ]),
    data = [
        ":setup",
        ":version",
        "//tf_blade",
        "//tf_blade/gpu:init",
        "//tf_blade/common:init",
        "//tf_blade/util:init",
    ],
)

sh_binary(
    name = "develop_pip_package",
    srcs = ["build_pip_package.sh"],
    args = [
        "--py_bin",
    ] + [
        python_bin_path(),
    ] + [
        "--device",
    ] + device_name() + [
        "--develop",
    ],
    data = [
        ":setup",
        ":version",
        "//tf_blade",
        "//tf_blade/gpu:init",
        "//tf_blade/common:init",
        "//tf_blade/util:init",
    ],
)
