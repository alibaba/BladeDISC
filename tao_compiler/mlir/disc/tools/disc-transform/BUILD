load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "get_compatible_with_cloud",
)

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "gentbl_filegroup", "td_library")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    packages = [
        "//tensorflow/compiler/mlir/...",
        "//tensorflow/compiler/tf2xla/...",
        "//tensorflow/compiler/xla/...",
        "//tensorflow/compiler/...",
    ],
)

gentbl_cc_library(
    name = "TransformPassIncGen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=DISCTransform",
            ],
            "transforms/transform_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/transform_passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "pass_details",
    hdrs = [
        "transforms/PassDetail.h",
    ],
    visibility = [
        "//visibility:private",  # This target is a private detail of pass implementations
    ],
    deps = [
        ":TransformPassIncGen",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "legalize_lmhlo_fusion_to_linalg",
    srcs = ["transforms/legalize_lmhlo_fusion_to_linalg.cc"],
    deps = [
        ":pass_details",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "all_passes",
    hdrs = [
        "transforms/passes.h",
        "transforms/register_passes.h",
    ],
    visibility = [
        ":friends",
    ],
    deps = [
        ":legalize_lmhlo_fusion_to_linalg",
        "@llvm-project//mlir:Pass",
    ],
    alwayslink = 1,
)