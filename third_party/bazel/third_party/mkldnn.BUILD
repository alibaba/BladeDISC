load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")
load("@local_config_blade_disc_helper//:build_defs.bzl",
    "cc_bin_path",
    "cxx_bin_path",
    "cxx11_abi",
)

filegroup(
    name = "all_source_files",
    srcs = glob(["**"]),
    visibility = ["//visibility:private"],
)

cmake(
    name = "mkldnn",
    cache_entries = {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_VERBOSE_MAKEFILE": "ON",
        "CMAKE_C_COMPILER": cc_bin_path(),
        "CMAKE_CXX_COMPILER": cxx_bin_path(),
        "MKL_ROOT": "",
        "USE_CXX11_ABI": cxx11_abi(),
    },
    generate_crosstool_file=False, ## This makes sure we use cxx by cache_entries settings
    build_args = ["-j"],
    lib_source = ":all_source_files",
    out_lib_dir = "lib64", # default out_lib_dir value is lib
    out_shared_libs = [
        "libfalcon_conv.so",
        "libHIE_framework.so.2",
    ],
    alwayslink = 1,
    visibility = ["//visibility:public"],
)

genrule(
    name = "hie_files",
    srcs = [
        "@hie//:hie",
    ],
    outs = [
        "libfalcon_conv.so",
        "libHIE_framework.so.2",
        "include/hie/dim.h",
        "include/hie/enum.h",
        "include/hie/hie.h",
        "include/hie/iplugin.h",
        "include/hie/macro.h",
    ],
    cmd = "cp -r $(SRCS) $(@D)",
    visibility = ["//visibility:public"],
)

genrule(
    name = "hie_headers",
    srcs = [
        "@hie//:hie",
    ],
    outs = [
    ],
    cmd = "cp -r $(SRCS) $(@D)",
    visibility = ["//visibility:public"],
)
