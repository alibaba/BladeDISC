load("@rules_python//python:defs.bzl", "py_library")
load("@local_config_mkl//:build_defs.bzl", "if_mkl_enabled")
load("//:build_defs.bzl", "device_name")

package(default_visibility = ["//visibility:public"])

genrule(
    name = "cp_mkl_libs",
    srcs = [
        "@local_config_mkl//:lib/compiler/lib/intel64_lin/libiomp5.so",
    ],
    outs = [
        "libiomp5.so",
    ],
    cmd = "cp -r $(SRCS) $(@D)",
)

genrule(
    name = "cp_tf_blade",
    srcs = [
        "//src:_tf_blade.so",
        "//src:tf_blade",
    ],
    outs = [
        "_tf_blade.so",
        "libtf_blade.so",
    ],
    cmd = "cp -r $(SRCS) $(@D)",
)

py_library(
    name = "init",
    srcs = ["__init__.py"],
    data = [
        "_tf_blade.so",
        "libtf_blade.so",
    ] + if_mkl_enabled([
        "libiomp5.so",
    ]),
)

py_library(
    name = "tf_blade",
    deps = [
        "//tf_blade:init",
        "//tf_blade/common",
        "//tf_blade/gpu",
        "//tf_blade/util",
    ],
)
