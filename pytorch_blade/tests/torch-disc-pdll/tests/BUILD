load("//bazel/tests:glob_lit_test.bzl", "glob_lit_tests")

package(licenses = ["notice"])

glob_lit_tests(
    data = [":test_utilities"],
    driver = "@llvm-project//mlir:run_lit.sh",
    test_file_exts = ["mlir"],
    tests_dir = "tests/torch-disc-pdll/tests",
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    srcs = glob([
        "*.pdll",
    ]),
    data = [
        "lit.site.cfg.py",
        "//tests/torch-disc-pdll:torch-disc-pdll",
        "//bazel/tests:lit.cfg.py",
        "@llvm-project//llvm:FileCheck",
    ],
)

exports_files(["lit.site.cfg.py"])
