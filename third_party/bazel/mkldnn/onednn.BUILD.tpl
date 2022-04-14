load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")
load("@local_config_blade_disc_helper//:build_defs.bzl",
    "cc_bin_path",
    "cxx_bin_path",
)

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all_source_files",
    srcs = glob(["**"]),
    visibility = ["//visibility:private"],
)

cmake(
    name = "onednn",
    cache_entries = {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_VERBOSE_MAKEFILE": "ON",
        "CMAKE_C_COMPILER": cc_bin_path(),
        "CMAKE_CXX_COMPILER": cxx_bin_path(),
        "CC": cc_bin_path(),
        "CXX": cxx_bin_path(),
        %{CXX11_SETTING}
        %{ACL_SETTING}
    },
    env = {
        %{ACL_ROOT}
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
