load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "gentbl_filegroup", "td_library")
load(
    "@org_tensorflow//tensorflow/tsl/platform:build_config.bzl",
    "tf_proto_library",
)
package(
    default_visibility = [
        "//visibility:public",
    ],
)

gentbl_cc_library(
    name = "TritonDISCPassIncGen",
    strip_include_prefix = "",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=triton_disc",
            ],
            "include/triton_disc/Conversion/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/triton_disc/Conversion/Passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
        "@llvm-project//mlir:NVVMToLLVMIRTranslation",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:MemRefDialect",
    ],
)

cc_library(
    name = "tritongpu-to-llvm-translation",
    includes = [
        "include"
    ],
    srcs = [
        "lib/DISC/TritonGPUToLLVMTranslation.cc",
        "lib/Conversion/TritonGPUToMLIRGPU.cc",
    ],
    hdrs = [
        "include/triton_disc/Conversion/Passes.h",
        "include/triton_disc/TritonGPUToLLVMTranslation.h",
        "include/triton_disc/Conversion/TritonGPUToMLIRGPU.h",
    ],
    deps = [
        ":TritonDISCPassIncGen",
        "@org_triton//:TritonDialect",
        "@org_triton//:TritonGPUIR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUPassIncGen",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:AllPassesAndDialects",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "triton-disc-opt",
    srcs = [
        "bin/triton_disc_opt.cc",
        "lib/Conversion/TritonGPUToMLIRGPU.cc",
    ],
    deps = [
        ":tritongpu-to-llvm-translation",
        "@com_google_protobuf//:protobuf",
        "@org_triton//:TritonGPUIR",
        "@org_triton//:TritonDialectCombine",
        "@org_triton//:TritonConversionPasses",
        "@org_triton//:TritonAnalysis",
        "@org_triton//:TritonDialectsAndPasses",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:LLVMDialect",
        "@org_disc_compiler//mlir/disc:disc_gpu_kernel_to_blob",
        "@org_disc_compiler//mlir/disc/tools/disc-transform:all_passes",
        "@org_disc_compiler//mlir/disc:disc_ral",
        "@org_disc_compiler//mlir/disc:all_passes",
        "@org_tensorflow//tensorflow/compiler/xla:xla_proto_cc",
        "@org_tensorflow//tensorflow/core/platform:protobuf",
        "@org_tensorflow//tensorflow/tsl/platform:env_impl",
        "@org_tensorflow//tensorflow/compiler/xla:debug_options_flags",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:all_passes",
        "@org_tensorflow//tensorflow/compiler/xla:execution_options_util",
        "@org_tensorflow//tensorflow/compiler/xla:xla_data_proto_cc",
        "@org_tensorflow//tensorflow/tsl/protobuf:histogram_proto_cc",
        "@org_tensorflow//tensorflow/tsl/profiler/utils:time_utils_impl",
        "@org_tensorflow//tensorflow/tsl/profiler/backends/cpu:host_tracer_utils",
        "@org_tensorflow//tensorflow/tsl/profiler/backends/cpu:traceme_recorder_impl",
    ],
)