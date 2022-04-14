package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tf_header_lib",
    hdrs = glob(["include/*.h"]),
    includes = ["include"],
)

cc_library(
    name = "libtensorflow_framework",
    srcs = glob(["lib/libtensorflow_framework.so.*"]),
    linkstatic=1,
)
