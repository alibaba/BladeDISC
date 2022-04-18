load("@rules_foreign_cc//foreign_cc:defs.bzl", "make")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all_source_files",
    srcs = glob(["**"]),
    visibility = ["//visibility:private"],
)

make(
    name = "acl",
    env = {
        "np": "24",
        "os": "linux",
        "arch": "arm64-v8a",  # (TODO): Make this configurable
    },
    lib_source = ":all_source_files",
    out_lib_dir = "build",
    out_include_dir = "",
    out_static_libs = [
        "libarm_compute.a",
        "libarm_compute_core.a",
        "libarm_compute_graph.a",
    ],
)

exports_files(["build", "Makefile"])
