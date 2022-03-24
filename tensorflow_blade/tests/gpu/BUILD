load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "tf_lstm_element_wise_ops_test",
    srcs = ["tf_lstm_element_wise_ops_test.py"],
    deps = [
        "//:context",
        "//tf_blade/util:tf_util",
        "//tf_blade/gpu",
    ],
    tags = [
        "gpu",
        "tf1",
    ],
)

py_test(
    name = "tf_non_max_suppression_opt_test",
    srcs = ["tf_non_max_suppression_opt_test.py"],
    deps = [
        "//:context",
        "//tf_blade/util:tf_util",
        "//tf_blade/gpu",
    ],
    tags = [
        "gpu",
    ],
)

py_test(
    name = "tf_to_trt_test",
    srcs = ["tf_to_trt_test.py"],
    deps = [
        "//:context",
        "//tf_blade/util:tf_util",
        "//tf_blade/gpu",
        "//tf_blade:init",
        "//tests:tf_test_common",
    ],
    tags = [
        "gpu",
    ],
)
