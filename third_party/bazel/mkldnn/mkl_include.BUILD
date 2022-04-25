package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mkl_include",
    hdrs = glob(["include/**"]),
    includes = ["include"],
    alwayslink = 1,  # targets depending on it should carry all symbols in its children.
    linkstatic = 1,
)
