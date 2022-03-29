load("@rules_python//python:defs.bzl", "py_test")

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
