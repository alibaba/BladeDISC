package(default_visibility = ["//visibility:public"])

exports_files(["lib"])

cc_library(
    name = "mkl_static_lib",
    srcs = glob(["lib/libmkl*.a"]),
    alwayslink = 1,  # targets depending on it should carry all symbols in its children.
    linkstatic = 1,
)
