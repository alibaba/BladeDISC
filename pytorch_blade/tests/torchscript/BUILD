load("//bazel/tests:glob_lit_test.bzl", "glob_lit_tests")
load("@rules_cc//cc:defs.bzl", "cc_binary")
load(
    "//bazel:build_defs.bzl",
    "if_quantization_enabled"
)

package(licenses = ["notice"])

cc_binary(
    name = "shape_analysis_tool",
    srcs = [
        "shape_analysis_tool.cpp"
    ],
    linkopts = [
        "-lm",
    ],
    deps = [
        "//pytorch_blade/compiler/jit:aten_custom_ops",
        "//pytorch_blade/common_utils:torch_blade_utils",
        "//pytorch_blade/compiler/jit/torch:shape_analysis",
        "@local_org_torch//:libtorch",
    ] + if_quantization_enabled([
        "//pytorch_blade/quantization:quantization_op",
    ]),
)

glob_lit_tests(
    data = [":test_utilities"],
    driver = "@llvm-project//mlir:run_lit.sh",
    test_file_exts = ["graph"],
    tests_dir = "tests/torchscript",
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "lit.site.cfg.py",
	":shape_analysis_tool",
        "//bazel/tests:lit.cfg.py",
        "@llvm-project//llvm:FileCheck",
    ],
)
exports_files(["lit.site.cfg.py"])
