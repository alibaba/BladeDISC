package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD/MIT-like license (for zlib)

cc_library(
    name = "aliyun_log_sdk",
    srcs = glob([
        "src/*.h",
        "src/*.c",
    ]),
    hdrs = glob(["src/*.h"]),
    includes = ["src"],
    deps = ["@platform_alibaba_curl//:curl"],
)
