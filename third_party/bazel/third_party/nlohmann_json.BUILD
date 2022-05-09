package(default_visibility = ["//visibility:public"])

load("@rules_cc//cc:defs.bzl", "cc_library")

licenses(["notice"])  # 3-Clause BSD

exports_files(["LICENSE.MIT"])

# cc_library(
#     name = "nlohmann_json",
#     hdrs = glob([
#         "include/**/*.hpp",
#     ]),
#     includes = ["include"],
#     visibility = ["//visibility:public"],
#     alwayslink = 1,
# )

cc_library(
    name = "nlohmann_json",
    hdrs = ["single_include/nlohmann/json.hpp"],
    strip_include_prefix = "single_include/",
)
