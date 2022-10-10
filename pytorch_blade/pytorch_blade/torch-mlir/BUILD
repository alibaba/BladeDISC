# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "gentbl_filegroup", "td_library")

package(
    default_visibility = [
    "//visibility:public",
    ],
)

# MhloConversion
td_library(
    name = "TorchMLIRConversionMhloPassesTdFiles",
    includes = ["include"],
    srcs = [
        "include/torch-mlir/Conversion/MhloPasses.td"
    ]
)

gentbl_cc_library(
    name = "TorchMLIRConversionMhloPassesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-pass-decls"],
            "include/torch-mlir/Conversion/MhloPasses.h.inc",
        )
    ],
    td_file = "include/torch-mlir/Conversion/MhloPasses.td",
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    deps = [
        ":TorchMLIRConversionMhloPassesTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ]
)

cc_library(
    name = "DiscTorchMLIRTorchToMhlo",
    srcs = [
        "lib/Conversion/MhloPasses.cpp",
        "lib/Conversion/TorchToMhlo/DiscTorchToMhlo.cpp",
        "lib/Dialect/TorchConversion/Transforms/VerifyMhloBackendContract.cpp",
        "lib/Dialect/TorchConversion/Transforms/ApplyValueSemantics.cpp",
        "lib/Dialect/TorchConversion/Transforms/ReduceTensorConversions.cpp",
        "lib/Dialect/TorchConversion/Transforms/DiscDecomposeComplexOps.cpp",
        "lib/Dialect/TorchConversion/Transforms/DiscConvertTorchToDiscMhlo.cpp",
    ],
    hdrs = [
        "include/torch-mlir/Conversion/MhloPasses.h",
    ],
    strip_include_prefix = "include",
    deps = [
        ":TorchMLIRConversionMhloPassesIncGen",
        "@mlir-hlo//:mlir_hlo",
        "@org_tensorflow//tensorflow/compiler/mlir/disc:mhlo_disc",
        "@llvm-project//mlir:Dialect",
        "@torch-mlir//:TorchMLIRTorchBackendTypeConversion",
        "@torch-mlir//:TorchMLIRTorchConversionDialect",
        "@torch-mlir//:TorchMLIRTorchConversionPasses",
        "@torch-mlir//:TorchMLIRTorchConversionPassesIncGen",
        "@torch-mlir//:TorchMLIRConversionPassesIncGen",
    ]
)
