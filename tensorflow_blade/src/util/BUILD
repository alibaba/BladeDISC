load("@local_config_tf//:build_defs.bzl", "tf_copts")
load("//:build_defs.bzl", "blade_cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tf_allocator_util",
    srcs = [
        "tf_allocator_util.cc",
    ],
    hdrs = [
        "tf_allocator_util.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,  # this target only allow static linking, no .so file generated.
    deps = [
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,  # targets depending on it should carry all symbols in its children.
)

cc_library(
    name = "logging",
    hdrs = [
        "logging.h",
    ],
    deps = [
        "@local_config_tf//:tf_header_lib",
    ],
)

blade_cc_test(
    name = "logging_test",
    srcs = [
        "logging_test.cc"
    ],
    deps = [
        ":logging",
        "@local_config_tf//:libtensorflow_framework",
    ]
)

blade_cc_test(
    name = "tf_allocator_util_test",
    srcs = [
        "tf_allocator_util_test.cc",
    ],
    deps = [
        ":tf_allocator_util",
        "@local_config_tf//:libtensorflow_framework",
    ],
    tags = ['gpu']
)
