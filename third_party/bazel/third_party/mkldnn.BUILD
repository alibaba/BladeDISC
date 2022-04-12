load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")
load("@local_config_blade_disc_helper//:build_defs.bzl",
    "cc_bin_path",
    "cxx_bin_path",
    "if_cxx11_abi",
    "if_disc_aarch64",
)

filegroup(
    name = "all_source_files",
    srcs = glob(["**"]),
    visibility = ["//visibility:private"],
)

# "ACL_ROOT_DIR": if_disc_aarch64("", "")[0],
# "DNNL_AARCH64_USE_ACL": if_disc_aarch64("ON", "")[0],
# "USE_CXX11_ABI": if_cxx11_abi("ON", "OFF"),
cmake(
    name = "onednn",
    cache_entries = {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_VERBOSE_MAKEFILE": "ON",
        "CMAKE_C_COMPILER": cc_bin_path(),
        "CMAKE_CXX_COMPILER": cxx_bin_path(),
        "USE_CXX11_ABI": "OFF",
        "CC": cc_bin_path(),
        "CXX": cxx_bin_path(),
    },
    generate_crosstool_file=False, ## This makes sure we use cxx by cache_entries settings
    build_args = ["-j"],
    lib_source = ":all_source_files",
    out_lib_dir = "lib64", # default out_lib_dir value is lib
    out_static_libs = [
        "libdnnl.a",
    ],
    alwayslink = 1,
    visibility = ["//visibility:public"],
)

genrule(
    name = "onednn_lib",
    srcs = [
        ":onednn",
    ],
    outs = [
        "libdnnl.a",
    ],
    cmd = "cp -r $(SRCS) $(@D)",
    visibility = ["//visibility:public"],
)
