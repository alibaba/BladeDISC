load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

filegroup(
    name = "mhlo_sources",
    srcs = glob(["mlir/mhlo/builder/*.cpp"])
)

filegroup(
    name = "mhlo_headers",
    srcs = glob(["mlir/mhlo/builder/*.h"])
)

cc_library(
    name = "mhlo_library",
    srcs = [":mhlo_sources"],
    hdrs = [":mhlo_headers"],
    includes = ["."],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@org_tensorflow//tensorflow/compiler/mlir/disc:mlir_disc_builder.so",
        "@org_tensorflow//tensorflow/compiler/mlir/disc:disc_map_chlo_to_hlo_op",
        "@org_tensorflow//tensorflow/compiler/mlir/disc:mhlo_disc",
    ]
)
