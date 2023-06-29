load("//bazel/tests:glob_lit_test.bzl", "glob_lit_tests")

package(licenses = ["notice"])

glob_lit_tests(
    data = [":test_utilities"],
    driver = "@llvm-project//mlir:run_lit.sh",
    test_file_exts = ["mlir"],
    tests_dir = "tests/mhlo",
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "lit.site.cfg.py",
        "//tests/mhlo/torch-mlir-opt:torch-mlir-opt",
        "//bazel/tests:lit.cfg.py",
        "@llvm-project//llvm:FileCheck",
    ],
)
exports_files(["lit.site.cfg.py"])
