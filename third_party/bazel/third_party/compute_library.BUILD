filegroup(
    name = "all_source_files",
    srcs = glob(["**"]),
    visibility = ["//visibility:private"],
)

sh_binary(
    name = "compute_library",
    srcs = [
        "@org_third_party//bazel/third_party:compute_library.sh",
    ],
    args = [
        "--arch 'arm64-v8a'",
    ] + [
        "--build_neon",
    ],
    deps = [
        ":all_source_files",
    ],
    data = [
        ":all_source_files",
    ],
)
