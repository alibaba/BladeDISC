load("@local_config_tf//:build_defs.bzl", "tf_copts")
load("//:build_defs.bzl", "if_tf_unsupported", "if_tf_supported")
load("@local_config_blade_disc_helper//:build_defs.bzl", "if_tensorrt_enabled", "if_internal")

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
        "logging.h"
    ],
    copts = if_tf_unsupported([
        "-DUSE_INTERNAL_LOGGER",
    ]),
    deps = if_tf_supported([
        "@local_config_tf//:tf_header_lib",
    ]),
    visibility = ["//visibility:public"],
)
