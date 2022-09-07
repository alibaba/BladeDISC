load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "get_compatible_with_cloud",
    "tf_cc_binary",
    "tf_cc_shared_object",
    "tf_cc_test",
    "tf_copts",
    "tf_native_cc_binary",
    "transitive_hdrs",
)
load(
    "@org_tensorflow//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
    "if_llvm_aarch64_available",
    "if_llvm_system_z_available",
)
load("@org_tensorflow//tensorflow/core/platform:build_config_root.bzl", "if_static")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "gentbl_filegroup", "td_library")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")
load("//tensorflow/compiler/mlir/disc:disc.bzl",
     "disc_cc_library",
     "if_cuda_or_rocm",
)

config_setting(
    name = "is_platform_alibaba",
    define_values = {"is_platform_alibaba": "true"},
)

config_setting(
    name = "is_patine",
    define_values = {"is_patine": "true"},
)

config_setting(
    name = "is_blade_gemm",
    define_values = {"is_blade_gemm": "true"},
)

config_setting(
    name = "is_mkldnn",
    define_values = {"is_mkldnn": "true"},
)

config_setting(
    name = "disc_aarch64",
    define_values = {"disc_aarch64": "true"},
)

config_setting(
    name = "disc_x86",
    define_values = {"disc_x86": "true"},
)

config_setting(
    name = "is_torch_disc",
    define_values = {"is_torch_disc": "true"},
)

package(
    default_visibility = [":friends"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    packages = [
        "//babelfish/device/...",
        "//learning/brain/experimental/mlir/...",
        "//learning/brain/experimental/swift_mlir/...",
        "//learning/brain/google/xla/kernels/...",
        "//learning/brain/swift/swift_mlir/...",
        "//tensorflow/compiler/mlir/...",
        "//tensorflow/compiler/tf2xla/...",
        "//tensorflow/compiler/xla/...",
        "//tensorflow/compiler/...",
        "//third_party/iree/...",
        "//third_party/mlir_edge/...",
        "//third_party/tf_runtime/tools/tf_kernel_gen/...",
    ],
)

td_library(
    name = "disc_td_files",
    srcs = glob(["transforms/*.td"]) + [
        # TODO(gcmn): These should be encapsulate in a td_library.
        "@llvm-project//mlir:include/mlir/Interfaces/ControlFlowInterfaces.td",
        "@llvm-project//mlir:include/mlir/Interfaces/CopyOpInterface.td",
        "@llvm-project//mlir:include/mlir/Interfaces/InferTypeOpInterface.td",
        "@llvm-project//mlir:include/mlir/Interfaces/LoopLikeInterface.td",
        "@llvm-project//mlir:include/mlir/Interfaces/ViewLikeInterface.td",
        "@llvm-project//mlir:include/mlir/Dialect/Shape/IR/ShapeBase.td",
        "@llvm-project//mlir:include/mlir/Dialect/Shape/IR/ShapeOps.td",
    ],
    compatible_with = get_compatible_with_cloud(),
    includes = ["."],
    deps = [
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:LoopLikeInterfaceTdFiles",
        "@llvm-project//mlir:MemRefOpsTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "DiscPassIncGen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=DISC",
            ],
            "transforms/disc_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/disc_passes.td",
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
        ":disc_shape",
        ":DiscPassIncGen",
        ":MhloDiscPassIncGen",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUPassIncGen",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "cycle_detector",
    srcs = ["utils/cycle_detector.cc"],
    hdrs = ["utils/cycle_detector.h"],
    includes = ["include"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)

tf_cc_test(
    name = "cycle_detector_test",
    srcs = ["transforms/utils/cycle_detector_test.cc"],
    deps = [
        ":cycle_detector",
        # TODO(b/160617323): Decouple MLIR HLO from TensorFlow/XLA
        "//tensorflow/compiler/xla:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "disc_util",
    srcs = [
        "disc_util.cc",
    ],
    hdrs = [
        "disc_util.h",
    ],
    deps = [
        ":cycle_detector",
        ":placement_utils",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/core:lib",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Pass",
    ],
)

disc_cc_library(
    name = "disc_compiler",
    srcs = [
        "disc_compiler.cc",
    ],
    hdrs = [
        "disc_compiler.h",
    ],
    deps = [
        ":all_passes",
        ":disc_ral",
        "//tensorflow/compiler/xla/mlir_hlo:all_passes",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo_gpu",
        "//tensorflow/core/platform:status",
        "//tensorflow/compiler/mlir/tensorflow:compile_mlir_util",
        "//tensorflow/compiler/mlir/tensorflow:tf_dialect_passes",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:IR",
    ],
    alwayslink = 1,
)

cc_library(
    name = "ral_inject_execution_context",
    srcs = ["transforms/ral_inject_execution_context.cc"],
    hdrs = ["transforms/passes.h"],
    deps = [
        ":disc_ral",
        ":pass_details",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_to_llvm",
    srcs = ["transforms/disc_to_llvm.cc"],
    hdrs = [
        "transforms/passes.h",
        "transforms/rewriters.h",
    ],
    includes = [
        "tensorflow/compiler/xla/mlir_hlo/include",
        "."
    ],
    deps = [
        ":disc_ral",
        ":disc_to_llvm_common",
        ":pass_details",
        ":placement_utils",
        "//tensorflow/core:lib",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TransformUtils",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        # "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        #"@llvm-project//mlir:StandardOpsTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ToLLVMIRTranslation",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "split_large_ops",
    srcs = ["transforms/split_large_ops.cc"],
    hdrs = ["transforms/passes.h"],
    includes = [
        "tensorflow/compiler/xla/mlir_hlo/include",
        "."
    ],
    deps = [
        ":pass_details",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "placement_utils",
    srcs = ["transforms/placement_utils.cc"],
    hdrs = ["transforms/placement_utils.h"],
    includes = ["include"],
    deps = [
        ":disc_map_hlo_to_lhlo_op",
        ":lmhlo_disc",
        ":mhlo_disc",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/mlir_hlo:map_hlo_to_lhlo_op",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:LLVMDialect",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_lhlo_elemental_utils",
    srcs = ["transforms/lhlo_elemental_utils.cc"],
    hdrs = ["transforms/lhlo_elemental_utils.h"],
    deps = [
        ":codegen_utils",
        ":disc_shape_optimization_utils",
        ":disc_util",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/mlir_hlo:map_lmhlo_to_scalar_op",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "disc_lhlo_legalize_roots_to_loops",
    srcs = ["transforms/lhlo_legalize_roots_to_loops.cc"],
    deps = [
        ":codegen_utils",
        ":disc_lhlo_elemental_utils",
        ":disc_supported_list",
        ":disc_util",
        ":fusion_utils",
        ":input_inline_fusion_pattern",
        ":pass_details",
        ":placement_utils",
        "//tensorflow/compiler/xla/mlir_hlo:LmhloPassIncGen",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/mlir_hlo:map_lmhlo_to_scalar_op",
        "//tensorflow/core:lib",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorDialect",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_fuse_splat_const",
    srcs = ["transforms/disc_fuse_splat_const.cc"],
    deps = [
        ":pass_details",
        "//tensorflow/compiler/xla/mlir_hlo:LmhloPassIncGen",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "revise_args_for_static_rank",
    srcs = [
        "transforms/revise_args_for_static_rank.cc",
    ],
    hdrs = ["transforms/passes.h"],
    includes = [
        "tensorflow/compiler/xla/mlir_hlo/include",
        "."
    ],
    deps = [
        ":pass_details",
        ":placement_utils",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//llvm:Core",
    ],
    alwayslink = 1,
)

cc_library(
    name = "conv_rewriter",
    srcs = ["transforms/conv_rewriter.cc"],
    hdrs = ["transforms/passes.h"],
    includes = [
        "tensorflow/compiler/xla/mlir_hlo/include",
        "."
    ],
    deps = [
        "@com_google_absl//absl/strings",
        ":pass_details",
        ":placement_utils",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "dot_rewriter",
    srcs = ["transforms/dot_rewriter.cc"],
    hdrs = ["transforms/passes.h"],
    includes = [
        "tensorflow/compiler/xla/mlir_hlo/include",
        "."
    ],
    deps = [
        ":disc_util",
        ":pass_details",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "gpu_conv_padding",
    srcs = ["transforms/gpu_conv_padding.cc"],
    hdrs = ["transforms/passes.h"],
    includes = [
        "tensorflow/compiler/xla/mlir_hlo/include",
        "."
    ],
    deps = [
        ":pass_details",
        ":placement_utils",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:TensorDialect",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mhlo_mark_shape_calc",
    srcs = ["transforms/mhlo_mark_shape_calc.cc"],
    hdrs = ["transforms/passes.h"],
    includes = [
        "tensorflow/compiler/xla/mlir_hlo/include",
        "."
    ],
    deps = [
        ":mhlo_disc",
        ":pass_details",
        ":placement_utils",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "mhlo_placer",
    srcs = ["transforms/mhlo_placer.cc"],
    hdrs = ["transforms/passes.h"],
    includes = [
        "tensorflow/compiler/xla/mlir_hlo/include",
        "."
    ],
    deps = [
        ":pass_details",
        ":placement_utils",
        ":mhlo_disc",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "codegen_utils",
    srcs = [
        "transforms/codegen_utils.cc",
    ],
    hdrs = [
        "transforms/codegen_utils.h",
    ],
    includes = [
        "tensorflow/compiler/xla/mlir_hlo/include",
    ],
    deps = [
        ":disc_shape",
        ":disc_util",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFUtils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_specialize_fusion_with_speculation",
    srcs = [
        "transforms/disc_specialize_fusion_with_speculation.cc",
    ],
    hdrs = ["transforms/passes.h"],
    deps = [
        ":codegen_utils",
        ":disc_util",
        ":fusion_utils",
        ":pass_details",
        ":placement_utils",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/core:lib",
        "@llvm-project//llvm:Core",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "element_type_converter",
    srcs = ["transforms/element_type_converter.cc"],
    hdrs = ["transforms/passes.h"],
    includes = [
        "tensorflow/compiler/xla/mlir_hlo/include",
        "."
    ],
    deps = [
        ":pass_details",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_parallel_loop_collapsing",
    srcs = ["transforms/parallel_loop_collapsing.cc"],
    includes = [
        "tensorflow/compiler/xla/mlir_hlo/include",
        "."
    ],
    deps = [
        ":codegen_utils",
        ":fusion_utils",
        ":pass_details",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/core:lib",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_parallel_loop_tiling",
    srcs = ["transforms/parallel_loop_tiling.cc"],
    includes = [
        "tensorflow/compiler/xla/mlir_hlo/include",
        "."
    ],
    deps = [
        ":codegen_utils",
        ":fusion_utils",
        ":pass_details",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/core:lib",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_for_loop_unroll_interleave",
    srcs = ["transforms/disc_for_loop_unroll_interleave.cc"],
    includes = [
        "tensorflow/compiler/xla/mlir_hlo/include",
        "."
    ],
    deps = [
        ":codegen_utils",
        ":fusion_utils",
        ":pass_details",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/core:lib",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
    ],
    alwayslink = 1,
)

cc_library(
    name = "revise_gpu_kernel_outlining",
    srcs = ["transforms/revise_kernel_outlining.cc"],
    includes = [
        "tensorflow/compiler/xla/mlir_hlo/include",
        "."
    ],
    deps = [
        ":pass_details",
        ":placement_utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "shape_utils",
    srcs = ["transforms/shape_utils.cc"],
    hdrs = ["transforms/shape_utils.h"],
    deps = [
        ":disc_shape_optimization_utils",
        ":disc_util",
        ":mhlo_disc",
        ":pass_details",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "fusion_utils",
    srcs = [
        "transforms/fusion_utils.cc",
        "transforms/fusion_utils_stitch_gpu.cc"],
    hdrs = ["transforms/fusion_utils.h"],
    deps = [
        ":codegen_utils",
        ":disc_lhlo_elemental_utils",
        ":disc_shape_optimization_utils",
        ":disc_supported_list",
        ":disc_util",
        ":lmhlo_disc",
        ":placement_utils",
        ":shape_utils",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "disc_fusion",
    srcs = ["transforms/lhlo_fusion.cc"],
    deps = [
        ":disc_shape_optimization_utils",
        ":disc_util",
        ":fusion_utils",
        ":pass_details",
        ":placement_utils",
        ":cycle_detector",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/core:lib",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_convert_const_to_ral",
    srcs = [
        "transforms/disc_convert_const_to_ral.cc",
    ],
    hdrs = ["transforms/passes.h"],
    deps = [
        ":disc_ral",
        ":pass_details",
        ":placement_utils",
        "//tensorflow/compiler/mlir/xla/ral:compile_metadata",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/core/platform:env",
        "@llvm-project//llvm:Core",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_map_hlo_to_lhlo_op",
    hdrs = ["transforms/disc_map_hlo_to_lhlo_op.h"],
    deps = [
        ":mhlo_disc",
        ":lmhlo_disc",
    ],
)

cc_library(
    name = "disc_hlo_legalize_to_lhlo",
    srcs = ["transforms/disc_hlo_legalize_to_lhlo.cc"],
    hdrs = [
        "transforms/passes.h",
        "transforms/rewriters.h",
    ],
    deps = [
        ":mhlo_disc",
        ":lmhlo_disc",
        ":disc_map_hlo_to_lhlo_op",
        ":pass_details",
        ":rng_uniform_custom_call",
        ":qconv_custom_call",
        ":sparse_custom_call",
        ":topk_custom_call",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:ShapeTransforms",
        #"@llvm-project//mlir:StandardOpsTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_lower_to_library_call",
    srcs = ["transforms/disc_lower_to_library_call.cc"],
    hdrs = [
        "transforms/passes.h",
        "transforms/rewriters.h",
    ],
    deps = [
        ":codegen_utils",
        ":disc_util",
        ":disc_ral",
        ":fusion_utils",
        ":lmhlo_disc",
        ":pass_details",
        ":placement_utils",
        ":rng_uniform_custom_call",
        ":topk_custom_call",
        ":qconv_custom_call",
        ":sparse_custom_call",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TransformUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TensorTransforms",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_assign_memory_space",
    srcs = ["transforms/disc_assign_memory_space.cc"],
    hdrs = [
        "transforms/passes.h",
        "transforms/rewriters.h",
    ],
    deps = [
        ":mhlo_disc",
        ":lmhlo_disc",
        ":disc_map_hlo_to_lhlo_op",
        ":disc_util",
        ":pass_details",
        ":placement_utils",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

disc_cc_library(
    name = "disc_gpu_kernel_to_blob",
    srcs = [
        "transforms/disc_gpu_kernel_to_blob.cc",
    ],
    hdrs = [
        "transforms/PassDetail.h",
        "transforms/passes.h",
        "transforms/rewriters.h",
    ],
    includes = ["."],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":codegen_utils",
        ":pass_details",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TransformUtils",
        "@llvm-project//mlir:MathToLibm",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:ComplexToLLVM",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUToGPURuntimeTransforms",
        "@llvm-project//mlir:GPUToNVVMTransforms",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        #"@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:NVVMToLLVMIRTranslation",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ToLLVMIRTranslation",
        "@llvm-project//mlir:Transforms",
        "//tensorflow/compiler/xla/service/gpu/llvm_gpu_backend",
        "//tensorflow/compiler/xla/service/gpu:stream_executor_util",
        "//tensorflow/compiler/xla/service/gpu:target_constants",
        "//tensorflow/compiler/xla/service/gpu:gpu_asm_opts_util",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/stream_executor/gpu:asm_compiler",
        "//tensorflow/core/platform:cuda_libdevice_path",
        "//tensorflow/core:lib",
    ] + if_cuda_is_configured([
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_asm_compiler",
    ]) + if_rocm_is_configured([
        "//tensorflow/core/platform:rocm_rocdl_path",
    ]),
)

cc_library(
    name = "disc_shape_to_std",
    srcs = ["transforms/disc_shape_to_std.cc"],
    hdrs = [
        "transforms/passes.h",
        "transforms/rewriters.h",
    ],
    deps = [
        ":pass_details",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)


cc_library(
    name = "disc_tensor_to_std",
    srcs = ["transforms/disc_tensor_to_std.cc"],
    hdrs = [
        "transforms/passes.h",
        "transforms/rewriters.h",
    ],
    deps = [
        ":pass_details",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_supported_list",
    hdrs = ["transforms/disc_supported_list.h.inc"],
)

cc_library(
    name = "input_inline_fusion_pattern",
    srcs = ["transforms/input_inline_fusion_pattern.cc"],
    hdrs = ["transforms/input_inline_fusion_pattern.h"],
    includes = ["tensorflow/compiler/xla/mlir_hlo/include"],
    deps = [
        ":codegen_utils",
        ":disc_lhlo_elemental_utils",
        ":fusion_utils",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/mlir_hlo:map_lmhlo_to_scalar_op",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "input_inline_fusion",
    srcs = ["transforms/input_inline_fusion_pass.cc"],
    deps = [
        ":disc_lhlo_elemental_utils",
        ":disc_supported_list",
        ":fusion_utils",
        ":input_inline_fusion_pattern",
        ":pass_details",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/mlir_hlo:map_lmhlo_to_scalar_op",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_remove_shape_constraints",
    srcs = ["transforms/remove_shape_constraints.cc"],
    hdrs = [
        "transforms/passes.h",
        "transforms/rewriters.h",
    ],
    deps = [
        ":pass_details",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:ShapeTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_hlo_to_std",
    srcs = ["transforms/disc_hlo_to_std.cc"],
    hdrs = [
        "transforms/passes.h",
        "transforms/rewriters.h",
    ],
    deps = [
        ":pass_details",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_std_bufferize",
    srcs = ["transforms/disc_std_bufferize.cc"],
    hdrs = [
        "transforms/passes.h",
        "transforms/rewriters.h",
    ],
    deps = [
        ":pass_details",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_memref_cse",
    srcs = ["transforms/disc_memref_cse.cc"],
    hdrs = [
        "transforms/passes.h",
    ],
    deps = [
        ":pass_details",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_memref_load_store_simplifier",
    srcs = ["transforms/disc_memref_load_store_simplifier.cc"],
    hdrs = [
        "transforms/passes.h",
    ],
    deps = [
        ":disc_util",
        ":pass_details",
        ":placement_utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

filegroup(
    name = "disc_lower_tf_ops_td_files",
    srcs = [
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@llvm-project//mlir:include/mlir/Interfaces/InferTypeOpInterface.td",
        "@llvm-project//mlir:include/mlir/Interfaces/LoopLikeInterface.td",
    ],
)

gentbl_cc_library(
    name = "disc_lower_tf_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/lower_tf.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/lower_tf.td",
    td_srcs = [
        ":disc_lower_tf_ops_td_files",
        #"@llvm-project//mlir:StdOpsTdFiles",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_optimize_td_files",
    ],
    deps = ["//tensorflow/compiler/xla/mlir_hlo:hlo_ops_td_files"],
)

cc_library(
    name = "disc_lower_tf",
    srcs = ["transforms/lower_tf.cc"],
    hdrs = [
        "transforms/passes.h",
        "transforms/rewriters.h",
    ],
    deps = [
        ":disc_lower_tf_inc_gen",
        ":disc_util",
        ":mhlo_disc",
        ":pass_details",
        ":rng_uniform_custom_call",
        ":topk_custom_call",
        ":qconv_custom_call",
        ":sparse_custom_call",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:convert_tensor",
        "//tensorflow/compiler/mlir/xla:legalize_utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_lower_gpu_ops_to_nvvm_ops",
    srcs = ["transforms/disc_lower_gpu_ops_to_nvvm_ops.cc"],
    includes = [
        "tensorflow/compiler/xla/mlir_hlo/include",
        "."
    ],
    deps = [
        ":disc_lower_gpu_ops_common",
        ":pass_details",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUToGPURuntimeTransforms",
        "@llvm-project//mlir:GPUToNVVMTransforms",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_llvm_insert_value_simplifier",
    srcs = ["transforms/disc_llvm_insert_value_simplifier.cc"],
    includes = [
        "tensorflow/compiler/xla/mlir_hlo/include",
        "."
    ],
    deps = [
        ":pass_details",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_function_dead_argument_elimination",
    srcs = ["transforms/disc_function_dead_argument_elimination.cc"],
    includes = [
        "tensorflow/compiler/xla/mlir_hlo/include",
        "."
    ],
    deps = [
        ":pass_details",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_side_effect_loop_invariant_code_motion",
    srcs = ["transforms/disc_side_effect_loop_invariant_code_motion.cc"],
    includes = [
        "tensorflow/compiler/xla/mlir_hlo/include",
        "."
    ],
    deps = [
        ":lmhlo_disc",
        ":pass_details",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_to_llvm_common",
    srcs = ["transforms/disc_to_llvm_common.cc"],
    includes = ["transforms/disc_to_llvm_common.h"],
    deps = [
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_lower_gpu_ops_common",
    srcs = ["transforms/disc_lower_gpu_ops_common.cc"],
    includes = ["transforms/disc_lower_gpu_ops_common.h"],
    deps = [
        ":disc_to_llvm_common",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_lower_gpu_ops_to_rocdl_ops",
    srcs = ["transforms/disc_lower_gpu_ops_to_rocdl_ops.cc"],
    includes = [
        "tensorflow/compiler/xla/mlir_hlo/include",
        "."
    ],
    deps = [
        ":disc_lower_gpu_ops_common",
        ":pass_details",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUToGPURuntimeTransforms",
        "@llvm-project//mlir:GPUToROCDLTransforms",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ROCDLToLLVMIRTranslation",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_remove_dead_buffer",
    srcs = ["transforms/disc_remove_dead_buffer.cc"],
    hdrs = [
        "transforms/passes.h",
        "transforms/rewriters.h",
    ],
    deps = [
        ":disc_util",
        ":pass_details",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_assign_kernel_name",
    srcs = ["transforms/disc_assign_kernel_name.cc"],
    hdrs = [
    ],
    deps = [
        ":fusion_utils",
        ":pass_details",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_unhandled_atomic_rmw_converter",
    srcs = ["transforms/disc_unhandled_atomic_rmw_converter.cc"],
    hdrs = ["transforms/passes.h"],
    includes = [
        "tensorflow/compiler/xla/mlir_hlo/include",
        "."
    ],
    deps = [
        ":pass_details",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:Pass",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:MemRefDialect",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_shape_simplifier",
    srcs = ["transforms/disc_shape_simplifier.cc"],
    hdrs = [
    ],
    deps = [
        ":mhlo_disc",
        ":pass_details",
        ":shape_utils",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_dot_merge",
    srcs = ["transforms/disc_dot_merge.cc"],
    hdrs = [
        "transforms/passes.h",
    ],
    deps = [
        ":cycle_detector",
        ":disc_util",
        ":mhlo_disc",
        ":pass_details",
        ":shape_utils",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_math_approximation",
    srcs = [
        "transforms/disc_math_approximation.cc",
    ],
    hdrs = ["transforms/passes.h"],
    deps = [
        ":pass_details",
        "@llvm-project//llvm:Core",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_flatten_memref_access",
    srcs = [
        "transforms/disc_flatten_memref_access.cc",
    ],
    hdrs = ["transforms/passes.h"],
    deps = [
        ":pass_details",
        ":codegen_utils",
        ":disc_lhlo_elemental_utils",
        ":disc_shape",
        "@llvm-project//llvm:Core",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_canonicalizer",
    srcs = [
        "transforms/disc_canonicalizer.cc",
    ],
    hdrs = ["transforms/passes.h"],
    deps = [
        ":pass_details",
        "@llvm-project//llvm:Core",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_memref_canonicalizer",
    srcs = ["transforms/disc_memref_canonicalizer.cc"],
    hdrs = [
    ],
    deps = [
        ":pass_details",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_outline_cpu_kernel",
    srcs = ["transforms/disc_outline_cpu_kernel.cc"],
    hdrs = [
    ],
    deps = [
        ":codegen_utils",
        ":disc_ral",
        ":fusion_utils",
        ":pass_details",
        ":placement_utils",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_cpu_map_parallel_loop",
    srcs = ["transforms/disc_cpu_map_parallel_loop.cc"],
    hdrs = [
    ],
    deps = [
        ":codegen_utils",
        ":disc_ral",
        ":fusion_utils",
        ":pass_details",
        ":placement_utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_stitch_fusion",
    srcs = ["transforms/disc_stitch_fusion.cc"],
    deps = [
        ":codegen_utils",
        ":disc_lhlo_elemental_utils",
        ":disc_util",
        ":fusion_utils",
        ":pass_details",
        ":placement_utils",
        "//tensorflow/compiler/xla/mlir_hlo:LmhloPassIncGen",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_transpose_simplifier",
    srcs = ["transforms/disc_transpose_simplifier.cc"],
    deps = [
        ":disc_util",
        ":mhlo_disc",
        ":pass_details",
        ":placement_utils",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lhlo_fusion_inliner",
    srcs = ["transforms/lhlo_fusion_inliner.cc"],
    deps = [
        ":pass_details",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "disc_algebra_simplifier",
    srcs = ["transforms/disc_algebra_simplifier.cc"],
    deps = [
        ":disc_shape_optimization_utils",
        ":mhlo_disc",
        ":pass_details",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        #"@llvm-project//mlir:StandardDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_dense_to_sparse",
    srcs = ["transforms/disc_dense_to_sparse.cc"],
    deps = [
        ":disc_shape_optimization_utils",
        ":mhlo_disc",
        ":pass_details",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_shape_optimization_utils",
    srcs = [
        "transforms/disc_shape_optimization_utils.cc"
    ],
    hdrs = [
        "transforms/disc_shape_optimization_utils.h",
    ],
    deps = [
        ":disc_shape",
        ":disc_util",
        ":pass_details",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_shape_optimization",
    srcs = [
        "transforms/disc_shape_optimization.cc"
    ],
    deps = [
        ":mhlo_disc",
        ":disc_shape_optimization_utils",
        ":disc_util",
        ":pass_details",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_strip_shape_constraint_ops",
    srcs = [
        "transforms/disc_strip_shape_constraint_ops.cc"
    ],
    deps = [
        ":disc_shape_optimization_utils",
        ":disc_util",
        ":pass_details",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_duplicate_computation_for_fusion",
    srcs = ["transforms/disc_duplicate_computation_for_fusion.cc"],
    deps = [
        ":disc_shape_optimization_utils",
        ":disc_util",
        ":fusion_utils",
        ":pass_details",
        ":placement_utils",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_convert_fake_quant_op",
    srcs = ["transforms/disc_convert_fake_quant_op.cc"],
    deps = [
        ":disc_shape_optimization_utils",
        ":mhlo_disc",
        ":pass_details",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_lower_quantize_and_dequantize",
    srcs = ["transforms/disc_lower_quantize_and_dequantize.cc"],
    deps = [
        ":disc_shape_optimization_utils",
        ":mhlo_disc",
        ":pass_details",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "all_passes",
    hdrs = [
        "transforms/register_passes.h",
    ],
    visibility = [
        ":friends",
    ],
    deps = [
        ":MhloDiscPassIncGen",
        ":conv_rewriter",
        ":disc_dot_merge",
        ":disc_algebra_simplifier",
        ":disc_assign_kernel_name",
        ":disc_assign_memory_space",
        ":disc_canonicalizer",
        ":disc_dense_to_sparse",
        ":disc_convert_const_to_ral",
        ":disc_convert_fake_quant_op",
        ":disc_cpu_map_parallel_loop",
        ":disc_duplicate_computation_for_fusion",
        ":disc_flatten_memref_access",
        ":disc_for_loop_unroll_interleave",
        ":disc_fuse_splat_const",
        ":disc_fusion",
        ":disc_hlo_legalize_to_lhlo",
        ":disc_hlo_to_std",
        ":disc_lhlo_legalize_roots_to_loops",
        ":disc_lower_tf",
        ":disc_lower_quantize_and_dequantize",
        ":disc_lower_to_library_call",
        ":disc_math_approximation",
        ":disc_memref_canonicalizer",
        ":disc_gpu_kernel_to_blob",
        ":disc_outline_cpu_kernel",
        ":disc_parallel_loop_collapsing",
        ":disc_parallel_loop_tiling",
        ":disc_remove_dead_buffer",
        ":disc_remove_shape_constraints",
        ":disc_shape_optimization",
        ":disc_shape_simplifier",
        ":disc_shape_to_std",
        ":disc_specialize_fusion_with_speculation",
        ":disc_std_bufferize",
        ":disc_stitch_fusion",
        ":disc_strip_shape_constraint_ops",
        ":disc_tensor_to_std",
        ":disc_to_llvm",
        ":disc_transpose_simplifier",
        ":dot_rewriter",
        ":element_type_converter",
        ":gpu_conv_padding",
        ":input_inline_fusion",
        ":lhlo_fusion_inliner",
        ":mhlo_mark_shape_calc",
        ":mhlo_placer",
        ":ral_inject_execution_context",
        ":revise_args_for_static_rank",
        ":revise_gpu_kernel_outlining",
        ":split_large_ops",
        ":disc_memref_cse",
        ":disc_memref_load_store_simplifier",
        ":disc_lower_gpu_ops_to_nvvm_ops",
        ":disc_lower_gpu_ops_to_rocdl_ops",
        ":disc_llvm_insert_value_simplifier",
        ":disc_function_dead_argument_elimination",
        ":disc_side_effect_loop_invariant_code_motion",
        ":disc_unhandled_atomic_rmw_converter",
        "@llvm-project//mlir:Pass",
    ],
    alwayslink = 1,
)

tf_cc_binary(
    name = "disc-opt",
    srcs = [
        "tools/disc-opt/disc-opt.cc",
    ],
    deps = [
        ":all_passes",
        ":disc_ral",
        ":mhlo_disc",
        ":lmhlo_disc",
        "//tensorflow/compiler/xla/mlir_hlo:all_passes",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo_gpu",
        "@boringssl//:crypto",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

gentbl_cc_library(
    name = "MhloDiscPassIncGen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=MhloDisc",
            ],
            "transforms/mhlo_disc_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/mhlo_disc_passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "mhlo_disc_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "IR/hlo_disc_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "IR/hlo_disc_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "IR/hlo_disc_ops.td",
    deps = ["//tensorflow/compiler/xla/mlir_hlo:hlo_ops_td_files"],
)

cc_library(
    name = "custom_call_base",
    srcs = [
        "IR/custom_call_base.cc",
    ],
    hdrs = [
        "IR/custom_call_base.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
    alwayslink = 1,
)

cc_library(
    name = "topk_custom_call",
    srcs = [
        "IR/topk_custom_call_op.cc",
    ],
    hdrs = [
        "IR/topk_custom_call_op.h",
    ],
    deps = [
        ":codegen_utils",
        ":custom_call_base",
        ":disc_ral",
        ":fusion_utils",
        ":lmhlo_disc",
        ":mhlo_disc",
        ":placement_utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
    alwayslink = 1,
)

cc_library(
    name = "qconv_custom_call",
    srcs = [
        "IR/qconv_custom_call_op.cc",
    ],
    deps = [
        ":codegen_utils",
        ":custom_call_base",
        ":disc_ral",
        ":disc_util",
        ":fusion_utils",
        ":lmhlo_disc",
        ":mhlo_disc",
        ":placement_utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:VectorDialect",
    ],
    alwayslink = 1,
)

cc_library(
    name = "rng_uniform_custom_call",
    srcs = [
        "IR/rng_uniform_custom_call_op.cc",
    ],
    hdrs = [
        "IR/rng_uniform_custom_call_op.h",
    ],
    deps = [
        ":codegen_utils",
        ":custom_call_base",
        ":disc_ral",
        ":lmhlo_disc",
        ":mhlo_disc",
        ":placement_utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
    alwayslink = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "sparse_custom_call",
    srcs = [
        "IR/sparse_custom_call_op.cc",
    ],
    deps = [
        ":custom_call_base",
        ":disc_ral",
        ":lmhlo_disc",
        ":mhlo_disc",
        ":placement_utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MathDialect",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mhlo_disc",
    srcs = [
        "IR/hlo_disc_ops.cc.inc",
        "IR/hlo_disc_ops.h.inc",
        "IR/hlo_disc_ops.cc",
    ],
    hdrs = [
        "IR/hlo_disc_ops.h",
    ],
    includes = ["include"],
    deps = [
        ":mhlo_disc_ops_inc_gen",
        ":custom_call_base",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:CopyOpInterface",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

gentbl_cc_library(
    name = "lmhlo_disc_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "IR/lhlo_disc_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "IR/lhlo_disc_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "IR/lhlo_disc_ops.td",
    deps = [
        "//tensorflow/compiler/xla/mlir_hlo:hlo_ops_td_files",
    ],
)

cc_library(
    name = "lmhlo_disc",
    srcs = [
        "IR/lhlo_disc_ops.cc.inc",
        "IR/lhlo_disc_ops.h.inc",
        "IR/lhlo_disc_ops.cc",
    ],
    hdrs = [
        "IR/lhlo_disc_ops.h",
    ],
    deps = [
        ":lmhlo_disc_ops_inc_gen",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo_ops_structs_inc_gen",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo_structured_interface",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

filegroup(
    name = "litfiles",
    srcs = glob(["runlit*py"]),
)

exports_files(["run_lit.sh"])

tf_cc_binary(
    name = "disc_compiler_main",
    srcs = ["disc_compiler_main.cc"],
    copts = tf_copts() + if_rocm_is_configured([
        "-DTENSORFLOW_USE_ROCM=1"
    ]),
    deps = [
        ":disc_compiler",
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service/gpu/llvm_gpu_backend",
        "//tensorflow/compiler/xla/stream_executor/lib",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:ARMCodeGen",  # fixdeps: keep
        "@llvm-project//llvm:CodeGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
        "@llvm-project//llvm:X86Disassembler",  # fixdeps: keep
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:ToLLVMIRTranslation",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cuda_driver",
    ]),
)

tf_cc_shared_object(
    name = "mlir_disc_builder.so",
    linkopts = select({
        "//conditions:default": [
            "-z defs",
            "-lm", "-Wl,-ldl",
            "-Wl,--version-script,$(location :mhlo_disc_builder.lds)",
        ],
    }),
    deps = [
        ":mhlo_disc",
        ":rng_uniform_custom_call",
        ":topk_custom_call",
        ":qconv_custom_call",
        ":sparse_custom_call",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
    ],
    data = [
        ":mhlo_disc_builder.lds"
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "disc_map_chlo_to_hlo_op",
    hdrs = ["transforms/disc_map_chlo_to_hlo_op.h"],
    deps = [
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
    ],
    visibility = ["//visibility:public"],
)

transitive_hdrs(
    name = "mlir_disc_headers",
    deps = [
        ":disc_map_chlo_to_hlo_op",
        ":mhlo_disc",
        "//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "//tensorflow/compiler/mlir/xla/ral:ral_base_cpu_context_impl",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:IR",
    ] + if_cuda_or_rocm([
        "//tensorflow/compiler/mlir/xla/ral:ral_base_cuda_context_impl",
    ]),
    visibility = ["//visibility:public"],
)

genrule(
    name = "install_mlir_disc_headers",
    srcs = [
        ":mlir_disc_headers",
    ],
    outs = ["mlir_disc_include"],
    cmd = """
    mkdir -p $@
    for f in $(SRCS); do
      d="$${f%/*}"
      d="$${d#bazel-out/*/bin/}"
      mkdir -p "$@/$${d}"
      cp -f "$${f}" "$@/$${d}/"
    done
    """,
    tags = ["manual"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mlir_disc_headers_lib",
    hdrs = [":mlir_disc_headers"],
    visibility = ["//visibility:public"],
)

gentbl_cc_library(
    name = "disc_shape_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "IR/disc_shape_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "IR/disc_shape_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "IR/disc_shape_ops.td",
    deps = [
        "//tensorflow/compiler/xla/mlir_hlo:hlo_ops_td_files",
    ],
)

cc_library(
    name = "disc_shape",
    srcs = [
        "IR/disc_shape_ops.cc.inc",
        "IR/disc_shape_ops.h.inc",
        "IR/disc_shape_ops.cc",
    ],
    hdrs = [
        "IR/disc_shape_ops.h",
    ],
    deps = [
        ":disc_shape_ops_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

gentbl_cc_library(
    name = "disc_ral_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "IR/disc_ral_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "IR/disc_ral_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "IR/disc_ral_ops.td",
    deps = [
        "//tensorflow/compiler/xla/mlir_hlo:hlo_ops_td_files",
    ],
)

cc_library(
    name = "disc_ral",
    srcs = [
        "IR/disc_ral_ops.cc.inc",
        "IR/disc_ral_ops.h.inc",
        "IR/disc_ral_ops.cc",
    ],
    hdrs = [
        "IR/disc_ral_ops.h",
    ],
    deps = [
        ":disc_ral_ops_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)
