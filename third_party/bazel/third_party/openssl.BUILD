load("@rules_foreign_cc//foreign_cc:defs.bzl", "configure_make")

# Read https://wiki.openssl.org/index.php/Compilation_and_Installation

filegroup(
    name = "all",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

configure_make(
    name = "openssl",
    configure_command = "config",
    configure_in_place = True,
    configure_options = [
        "-fPIC",
        "no-shared",
        "no-unit-test",
        "enable-egd",
    ],
    args = ["-j"],
    targets = ["", "install_sw"],
    lib_source = "@openssl//:all",
    out_lib_dir = "lib",
    out_static_libs = [
        "libssl.a",
        "libcrypto.a",
    ],
    visibility = ["//visibility:public"],
)
