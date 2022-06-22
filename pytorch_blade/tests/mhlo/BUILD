load("//tests/mhlo/torch-mlir-opt:glob_lit_test.bzl", "glob_lit_tests")

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
    data = [
        "lit.cfg.py", "lit.site.cfg.py",
        "//tests/mhlo/torch-mlir-opt:torch-mlir-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)
exports_files(["lit.cfg.py", "lit.site.cfg.py"])
