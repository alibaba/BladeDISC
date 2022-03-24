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
    ],
)

py_library(
    name = "tf_lstm_element_wise_ops",
    srcs = [
        "tf_lstm_element_wise_ops.py",
    ],
    deps = [
        "//tf_blade/util:simple_graph",
        "//tf_blade/util:tf_graph_transform_util",
        "//tf_blade/util:tf_import_helper",
    ],
)

py_library(
    name = "tf_non_max_suppression_opt",
    srcs = [
        "tf_non_max_suppression_opt.py",
    ],
    deps = [
        "//tf_blade/util:tf_graph_transform_util",
        "//tf_blade/util:tf_import_helper",
        "//tf_blade/util:tf_util",
    ],
)

py_library(
    name = "gpu",
    deps = [
        ":tf_to_trt",
        ":tf_lstm_element_wise_ops",
        ":tf_non_max_suppression_opt",
    ],
)
