package(default_visibility = ["//visibility:public"])

exports_files(["build", "acl"])

cc_library(
    name = "onednn",
    hdrs = glob([
        "build/install/include/*",
        "build/intel/include/*",
    ]),
    includes = [
        "build/install/include",
        "build/intel/include",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "onednn_acl",
    hdrs = glob([
        "build/install/include/*",
        "build/intel/include/*",
        "build/acl/ComputeLibrary/arm_compute/**/*",
        "build/acl/ComputeLibrary/include/**/*",
    ]),
    includes = [
        "build/install/include",
        "build/intel/include",
        "build/acl/ComputeLibrary/",
        "build/acl/ComputeLibrary/include",
    ],
    visibility = ["//visibility:public"],
)
