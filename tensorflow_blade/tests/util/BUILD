load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "tf_util_test",
    srcs = ["tf_util_test.py"],
    deps = [
        "//:context",
        "//tf_blade/util:tf_util",
    ],
)

py_test(
    name = "simple_graph_test",
    srcs = ["simple_graph_test.py"],
    deps = [
        "//:context",
        "//tf_blade/util:simple_graph",
        "//tf_blade/util:tf_graph_transform_util",
    ],
)
