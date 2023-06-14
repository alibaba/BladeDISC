load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "get_compatible_with_cloud",
)

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "gentbl_filegroup", "td_library")

package(
    default_visibility = ["//visibility:public"],
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

td_library(
    name = "LinalgExtTdFiles",
    srcs = glob([
        "LinalgExt/*.td",
    ]),
    includes=["external/org_tensorflow"],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PDLDialectTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@llvm-project//mlir:TransformDialectTdFiles",
    ],
)

gentbl_cc_library(
    name = "DISCLinalgExtIncGen",
    tbl_outs = [
        (
            [
                "--dialect=disc_linalg_ext",
                "--gen-dialect-decls",
            ],
            "LinalgExt/LinalgExtDialect.h.inc",
        ),
        (
            [
                "--dialect=disc_linalg_ext",
                "--gen-dialect-defs",
            ],
            "LinalgExt/LinalgExtDialect.cc.inc",
        ),
        (
            ["--gen-op-decls"],
            "LinalgExt/LinalgExtOps.h.inc",
        ),
        (
            ["--gen-op-defs"],
            "LinalgExt/LinalgExtOps.cc.inc",
        ),
        (
            ["--gen-typedef-decls"],
            "LinalgExt/LinalgExtTypes.h.inc",
        ),
        (
            ["--gen-typedef-defs"],
            "LinalgExt/LinalgExtTypes.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "LinalgExt/LinalgExtOps.td",
    deps = [
        ":LinalgExtTdFiles",
        ":DISCLinalgExtEnumsIncGen",
        "@llvm-project//mlir:CallInterfacesTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:TilingInterfaceTdFiles",
        "@llvm-project//mlir:ViewLikeInterfaceTdFiles",
    ],
)

gentbl_cc_library(
    name = "DISCLinalgExtInterfacesIncGen",
    tbl_outs = [
        (
            ["--gen-op-interface-decls"],
            "LinalgExt/LinalgExtOpInterfaces.h.inc",
        ),
        (
            ["--gen-op-interface-defs"],
            "LinalgExt/LinalgExtOpInterfaces.cc.inc",
        ),
        (
            ["--gen-type-interface-decls"],
            "LinalgExt/LinalgExtTypeInterfaces.h.inc",
        ),
        (
            ["--gen-type-interface-defs"],
            "LinalgExt/LinalgExtTypeInterfaces.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "LinalgExt/LinalgExtInterfaces.td",
    deps = [
        ":LinalgExtTdFiles",
    ],
)

gentbl_cc_library(
    name = "DISCLinalgExtEnumsIncGen",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "LinalgExt/LinalgExtEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "LinalgExt/LinalgExtEnums.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "LinalgExt/LinalgExtEnums.td",
    deps = [":LinalgExtTdFiles"],
)

cc_library(
    name = "DISCLinalgExtDialect",
    srcs = glob([
        "LinalgExt/*.cc",
    ]),
    hdrs = glob([
        "LinalgExt/*.h",
    ]),
    deps = [
        ":DISCLinalgExtEnumsIncGen",
        ":DISCLinalgExtIncGen",
        ":DISCLinalgExtInterfacesIncGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithUtils",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TensorUtils",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
    alwayslink = 1,
)

gentbl_cc_library(
    name = "transform_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "TransformOps/TransformOpsExt.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "TransformOps/TransformOpsExt.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "TransformOps/TransformOpsExt.td",
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:CopyOpInterfaceTdFiles",
        "@llvm-project//mlir:LoopLikeInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@llvm-project//mlir:TransformDialectTdFiles",
        "@llvm-project//mlir:PDLDialectTdFiles",
    ],
)

cc_library(
    name = "disc_transform_utils",
    srcs = [
        "utils.cc",
    ],
    hdrs = [
        "utils.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_transform_ops",
    srcs = [
        "TransformOps/TransformOpsExt.cc",
        "TransformOps/TransformOpsExt.cc.inc",
        "TransformOps/TransformOpsExt.h.inc",
    ],
    hdrs = [
        "TransformOps/TransformOpsExt.h",
    ],
    deps = [
        ":DISCLinalgExtDialect",
        ":disc_transform_utils",
        ":transform_ops_inc_gen",
        "@iree-dialects//:IREELinalgExtDialect",
        "@iree-dialects//:IREELinalgExtTransformOps",
        "@iree-dialects//:IREELinalgTransformDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithUtils",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:DestinationStyleOpInterface",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LinalgTransformOps",
        "@llvm-project//mlir:LinalgUtils",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:PDLDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:ShapeTransforms",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TensorUtils",
        "@llvm-project//mlir:TransformDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorTransformOps",
    ],
    alwayslink = 1,
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
        ":DISCLinalgExtDialect",
        ":disc_transform_utils",
        ":pass_details",
        "//mlir/disc:disc_lhlo_elemental_utils",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:legalize_to_linalg_utils",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:lhlo",
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
    name = "transform_dialect_interpreter",
    srcs = ["transforms/transform_dialect_interpreter.cc"],
    deps = [
        ":DISCLinalgExtDialect",
        ":disc_transform_ops",
        ":pass_details",
        "@iree-dialects//:IREELinalgExtDialect",
        "@iree-dialects//:IREELinalgExtTransformOps",
        "@iree-dialects//:IREELinalgTransformDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LinalgTransformOps",
        "@llvm-project//mlir:PDLDialect",
        "@llvm-project//mlir:PDLInterpDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFTransformOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformDialect",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorDialect",
    ],
    alwayslink = 1,
)

cc_library(
    name = "rewrite-payload-ir-for-ral",
    srcs = ["transforms/rewrite_payload_ir_for_ral.cc"],
    deps = [
        ":pass_details",
        "//mlir/disc:placement_utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "memref_copy_to_linalg",
    srcs = ["transforms/memref_copy_to_linalg.cc"],
    deps = [
        ":pass_details",
        ":disc_transform_utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "convert-foreach-thread-op-to-parallel-op",
    srcs = ["transforms/convert_foreach_thread_op_to_parallel_op.cc"],
    deps = [
        ":pass_details",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SCFDialect",
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
    visibility = ["//visibility:public"],
    deps = [
        ":convert-foreach-thread-op-to-parallel-op",
        ":legalize_lmhlo_fusion_to_linalg",
        ":memref_copy_to_linalg",
        ":rewrite-payload-ir-for-ral",
        ":transform_dialect_interpreter",
        "@llvm-project//mlir:Pass",
    ],
    alwayslink = 1,
)