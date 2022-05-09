package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT/X derivative license

load("@rules_foreign_cc//foreign_cc:defs.bzl", "configure_make")

filegroup(
    name = "all",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

configure_make(
    name = "libuuid",
    configure_in_place = True,
    configure_options = ["--with-pic"],
    lib_source = "@libuuid//:all",
)
