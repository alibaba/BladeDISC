# Description: CUB library which is a set of primitives for GPU programming.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # BSD

cc_library(
    name = "cub",
    hdrs = glob(["cub/**"]),
    deps = ["@local_config_cuda//cuda:cuda_headers"],
)
