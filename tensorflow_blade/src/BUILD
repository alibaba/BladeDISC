load("@bazel_skylib//lib:selects.bzl", "selects")
load("@local_config_mkl//:build_defs.bzl", "if_mkl_enabled")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@local_config_tf//:build_defs.bzl", "tf_copts", "tf_version_define")
load("//:build_defs.bzl", "device_name")
load("@local_config_blade_helper//:build_defs.bzl", "python_bin_path")

package(default_visibility = ["//visibility:public"])

genrule(
    name = 'tf_version_header_genrule',
    outs = ["tf_version.h"],
    cmd = "touch $@;chmod +w $@;echo '%s' > $@" % tf_version_define(),
)

cc_library(
    name = "tf_version_header",
    srcs = ["tf_version.h"],
    data = [
        ':tf_version_header_genrule'
    ],
)

pybind_extension(
    name = "_tf_blade",
    srcs = [
        "pybind.cpp",
    ],
    linkopts = [
        "-Wl,-rpath,$$ORIGIN",
        "-Wl,-soname,_tf_blade.so",
    ],
    copts = tf_copts(),
    deps = [
        ":tf_version_header",
        "@local_config_tf//:libtensorflow_framework",
    ] + select({
        "@local_config_cuda//:is_cuda_enabled": [
            "@local_config_cuda//cuda:cudart_static",
            "@local_config_tensorrt//:nvinfer_static",
        ],
        "//conditions:default": [],
    }),
)
