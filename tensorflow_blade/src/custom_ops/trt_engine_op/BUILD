load("//:build_defs.bzl", "tf_blade_library")

package(default_visibility = ["//visibility:public"])

tf_blade_library(
    name = "trt_engine_op",
    srcs = ["trt_engine_op.cc"],
    deps = [
        "//src/tensorrt:tensorrt_bridge_impl",
    ],
)
