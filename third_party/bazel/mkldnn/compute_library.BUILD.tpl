package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all_source_files",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

sh_binary(
    name = "scons_build_acl",
    srcs = [
        "@local_config_mkldnn//:compute_library.sh",
    ],
    args = [
        "--arch 'arm64-v8a'",
    ] + [
        "--build_neon",
    ],
    data = [
        ":all_source_files",
    ],
)

exports_files(["build", "arm_compute", "include"])

cc_library(
    name = "arm_compute_lib",
    srcs = [
    ],
)

genrule(
    name = "arm_compute_lib_gen",
    srcs = [
        "build/libarm_compute.a",
    ],
    outs = [
        "libarm_compute.a",
    ],
    cmd = "pwd;ls -lh",
    visibility = ["//visibility:public"],
)

genrule(
    name = "graph_headers",
    srcs = [
        "arm_compute/graph.h",
    ],
    outs = [
        "graph.h",
    ],
    cmd = "pwd; cp -r $(SRCS) $(@D)",
    visibility = ["//visibility:public"],
)
