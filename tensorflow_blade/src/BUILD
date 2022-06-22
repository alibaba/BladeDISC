load("@bazel_skylib//lib:selects.bzl", "selects")
load("@local_config_mkl//:build_defs.bzl", "if_mkl_enabled")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@local_config_tf//:build_defs.bzl", "tf_copts", "tf_version")
load("@local_config_blade_disc_helper//:build_defs.bzl", "if_tensorrt_enabled", "if_platform_alibaba")

package(default_visibility = ["//visibility:public"])


cc_binary(
    name = "tf_blade",
    copts = tf_copts(),
    linkopts = [
        "-ldl",
        "-lm",
        "-Wl,-rpath,$$ORIGIN",
        "-Wl,--version-script,$(location :tf_blade.lds)",
        "-Wl,-soname,libtf_blade.so",
    ],
    linkshared = 1,
    linkstatic = 1,
    deps = [
        "@org_tao_bridge//:tao_bridge",
    ] + if_tensorrt_enabled([
        "//src/custom_ops/trt_engine_op",
        "//src/tensorrt:tensorrt_bridge_impl",
    ]) + if_platform_alibaba([
        "//src/internal:tf_blade_internal",
    ]),
    data = [
        ":tf_blade.lds"
    ]
)

##########################################################
cc_library(
    name = "input_version_header",
    hdrs = [
        "tf_compatible_version.h.in"
    ],
)

genrule(
    name = "tf_version_header_genrule",
    srcs = [
        ":input_version_header"
    ],
    outs = ["tf_compatible_version.h"],
    cmd = "cp src/tf_compatible_version.h.in $@;chmod +w $@;sed -i 's/TF_COMPATIBLE_STRING/%s/g' $@" % tf_version(),
)

cc_library(
    name = "tf_version_header",
    srcs = ["tf_compatible_version.h"],
    data = [
        ':tf_version_header_genrule'
    ],
)

pybind_extension(
    name = "_tf_blade",
    srcs = [
        "pybind.cpp",
        ":tf_blade",
    ] + if_tensorrt_enabled([
        "//src/tensorrt:pybind_functions.h",
        "//src/tensorrt:pybind_functions.cpp",
    ]) + if_platform_alibaba([
        "//src/internal:pybind_srcs"
    ]),
    linkopts = [
        "-Wl,-rpath,$$ORIGIN",
        "-Wl,-soname,_tf_blade.so",
    ],
    copts = if_tensorrt_enabled([
        "-DBLADE_WITH_TENSORRT",
    ]) + tf_copts(),
    deps = [
        ":tf_version_header",
        ":tf_blade",
        "@local_config_tf//:libtensorflow_framework",
    ],
)
