load("@org_tensorflow//tensorflow:tensorflow.bzl", "filegroup")
load("//mlir/disc:glob_lit_test.bzl", "glob_lit_tests")

package(licenses = ["notice"])

glob_lit_tests(
    data = [":test_utilities"],
    driver = "@llvm-project//mlir:run_lit.sh",
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    srcs = glob([
        "*.pdll",
    ]),
    data = [
        "//mlir/disc:disc-pdll",
        "@llvm-project//llvm:FileCheck",
    ],
)