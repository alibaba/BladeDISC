load("@local_config_tf//:build_defs.bzl", "tf_copts")

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
