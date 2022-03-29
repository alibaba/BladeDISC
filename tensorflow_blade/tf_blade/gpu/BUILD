load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "init",
    srcs = ["__init__.py"],
)

py_library(
    name = "tf_to_trt",
    srcs = [
        "tf_to_trt.py",
    ],
    deps = [
        "//tf_blade/util:simple_graph",
        "//tf_blade/util:tf_conversion_util",
        "//tf_blade/util:tf_graph_transform_util",
        "//tf_blade/util:tf_import_helper",
        "//tf_blade/util:tf2onnx_import_helper",
    ],
)

py_library(
    name = "gpu",
    deps = [
        ":tf_to_trt",
    ],
)
