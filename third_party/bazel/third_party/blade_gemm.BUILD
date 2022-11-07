load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")
load("@local_config_blade_disc_helper//:build_defs.bzl",
    "cc_bin_path",
    "cxx_bin_path",
    "blade_gemm_nvcc",
    "blade_gemm_nvcc_archs",
    "blade_gemm_library_kernels",
    "blade_gemm_tvm",
    "blade_gemm_rocm_path",
    "foreign_make_args",
)
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")

filegroup(
    name = "all_source_files",
    srcs = glob(["**"]),
    visibility = ["//visibility:private"],
)

cmake(
    name = "blade_gemm",
    cache_entries = {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_VERBOSE_MAKEFILE": "ON",
        "CMAKE_C_COMPILER": cc_bin_path(),
        "CMAKE_CXX_COMPILER": cxx_bin_path(),
        "USE_TVM": blade_gemm_tvm(),
        "ROCM_PATH": blade_gemm_rocm_path(),
    },
    env = {
        "CC": cc_bin_path(),
        "CXX": cxx_bin_path(),
        "CUDACXX": blade_gemm_nvcc(),
    },
    generate_crosstool_file=False, ## This makes sure we use cxx by cache_entries settings
    build_args = foreign_make_args(),
    lib_source = ":all_source_files",
    out_lib_dir = "lib64", # default out_lib_dir value is lib
    out_static_libs = [
        "libblade_gemm.a",
    ],
    alwayslink = 1,
    visibility = ["//visibility:public"],
)
