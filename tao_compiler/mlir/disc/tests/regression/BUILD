load("//mlir/disc/tests:glob_op_test.bzl", "glob_op_tests")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    packages = [
        "//babelfish/device/...",
        "//learning/brain/experimental/mlir/...",
        "//learning/brain/experimental/swift_mlir/...",
        "//learning/brain/google/xla/kernels/...",
        "//learning/brain/swift/swift_mlir/...",
        "//tensorflow/compiler/mlir/...",
        "//tensorflow/compiler/tf2xla/...",
        "//tensorflow/compiler/xla/...",
        "//tensorflow/compiler/...",
        "//third_party/iree/...",
        "//third_party/mlir_edge/...",
        "//third_party/tf_runtime/tools/tf_kernel_gen/...",
    ],
)

glob_op_tests(
    data = [],
    driver = "@llvm-project//mlir:run_lit.sh",
    test_file_exts = ["cc"],
)
