load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "init",
    srcs = ["__init__.py"],
)

py_library(
    name = "tf_import_helper",
    srcs = [
        "tf_import_helper.py",
    ],
)

py_library(
    name = "tf_util",
    srcs = [
        "tf_util.py",
    ],
    deps = [
        ":tf_import_helper",
    ],
)

py_library(
    name = "graph_transform",
    srcs = [
        "graph_transform.py",
    ],
    deps = [
        ":tf_util",
    ],
)

py_library(
    name = "simple_graph",
    srcs = [
        "simple_graph.py",
    ],
    deps = [
        ":tf_util",
        ":tf_import_helper",
    ],
)

py_library(
    name = "tf_conversion_util",
    srcs = [
        "tf_conversion_util.py",
    ],
    deps = [
        ":tf_import_helper",
    ],
)

py_library(
    name = "tf_graph_transform_util",
    srcs = [
        "tf_graph_transform_util.py",
    ],
    deps = [
        ":graph_transform",
        ":simple_graph",
        ":tf_util",
        ":tf_import_helper",
    ],
)

py_library(
    name = "tf_hierarchy_pattern_match",
    srcs = [
        "tf_hierarchy_pattern_match.py",
    ],
    deps = [
        ":simple_graph",
        ":tf_graph_transform_util",
        ":tf_import_helper",
    ],
)

py_library(
    name = "util",
    deps = [
        ":graph_transform",
        ":simple_graph",
        ":tf_conversion_util",
        ":tf_graph_transform_util",
        ":tf_import_helper",
        ":tf_util",
    ],
)
