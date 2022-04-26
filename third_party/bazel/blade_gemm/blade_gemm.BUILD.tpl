load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")
load("@local_config_blade_disc_helper//:build_defs.bzl",
    "cc_bin_path",
    "cxx_bin_path",
)

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
        "CUTLASS_NVCC_ARCHS": "%{CUTLASS_NVCC_ARCHS}",
        "CUTLASS_LIBRARY_KERNELS": "%{CUTLASS_LIBRARY_KERNELS}",
        "CUTLASS_ENABLE_TESTS": "OFF",
        "CUTLASS_UNITY_BUILD_ENABLED": "ON",
    },
    env = {
        "CC": cc_bin_path(),
        "CXX": cxx_bin_path(),
        "CUDACXX": "%{CUTLASS_CUDACXX}",
    },
    generate_crosstool_file=False, ## This makes sure we use cxx by cache_entries settings
    build_args = ["-j"],
    lib_source = ":all_source_files",
    out_lib_dir = "lib64", # default out_lib_dir value is lib
    out_static_libs = [
        "libblade_gemm.a",
    ],
    alwayslink = 1,
    visibility = ["//visibility:public"],
)
