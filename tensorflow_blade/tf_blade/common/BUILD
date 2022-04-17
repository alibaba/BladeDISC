load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "init",
    srcs = ["__init__.py"],
)

py_library(
    name = "tf_grappler",
    srcs = [
        "tf_grappler.py",
    ],
    deps = [
        "//tf_blade/util:tf_import_helper",
    ],
)

py_library(
    name = "common",
    deps = [
        ":tf_grappler",
    ],
)
