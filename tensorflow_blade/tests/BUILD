load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

genrule(
    name = "cp_tf_blade_libs",
    srcs = [
        "@org_tf_blade//src:tf_blade",
    ],
    outs = [
        "libtf_blade.so",
    ],
    cmd = "cp -r $(SRCS) $(@D)",
)

py_library(
    name = "tf_blade_ops_ut_common",
    srcs = ["tf_blade_ops_ut_common.py"],
    data = [
        "libtf_blade.so"
    ],
)
