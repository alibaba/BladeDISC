package(default_visibility = ["//visibility:public"])

exports_files(["build"])

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
    #deps = [":dnnl_static",],
    visibility = ["//visibility:public"],
)

