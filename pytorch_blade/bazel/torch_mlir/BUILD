load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "gentbl_filegroup", "td_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],
)

cc_library(
    name = "TorchMLIRCAPI",
    srcs = [
       "lib/CAPI/Dialects.cpp",
       "lib/CAPI/Registration.cpp",
       "lib/CAPI/TorchOps.cpp", 
       "lib/CAPI/TorchTypes.cpp",
    ],
    hdrs = [
    ],
    deps = [
        "@torch-mlir//:TorchMLIRInitAll",
        "@torch-mlir//:TorchMLIRTorchDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:IR"
    ]
)

cc_library (
  name = "jit_ir_importer",
  srcs =  [
    "python/torch_mlir/dialects/torch/importer/jit_ir/csrc/ivalue_importer.cpp",
    "python/torch_mlir/dialects/torch/importer/jit_ir/csrc/class_annotator.cpp",
    "python/torch_mlir/dialects/torch/importer/jit_ir/csrc/function_importer.cpp",
    "python/torch_mlir/dialects/torch/importer/jit_ir/csrc/node_importer.cpp",
    "python/torch_mlir/dialects/torch/importer/jit_ir/csrc/torch_to_mlir_utils.cpp",
  ],
  strip_include_prefix = "python/torch_mlir/dialects/torch/importer/jit_ir/csrc",
  hdrs = [
    "python/torch_mlir/dialects/torch/importer/jit_ir/csrc/mlir_utils.h",
    "python/torch_mlir/dialects/torch/importer/jit_ir/csrc/class_annotator.h",
    "python/torch_mlir/dialects/torch/importer/jit_ir/csrc/ivalue_importer.h",
    "python/torch_mlir/dialects/torch/importer/jit_ir/csrc/function_importer.h",
    "python/torch_mlir/dialects/torch/importer/jit_ir/csrc/node_importer.h",
    "python/torch_mlir/dialects/torch/importer/jit_ir/csrc/import_options.h",
    "python/torch_mlir/dialects/torch/importer/jit_ir/csrc/torch_to_mlir_utils.h",
  ],
  deps = [
    ":TorchMLIRCAPI",
    "@local_org_torch//:libtorch",
  ],
)
