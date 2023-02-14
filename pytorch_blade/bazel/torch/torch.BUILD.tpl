package(default_visibility = ["//visibility:public"])

load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
cc_library(
    name = "libtorch",
    srcs = [
            "lib/libtorch.so",
            "lib/libtorch_cpu.so",
            "lib/libtorch_global_deps.so",
        ] + glob(
            ["lib/libtorch_cuda_cu.so", "lib/libtorch_cuda_cpp.so", "lib/libgomp*.so.1"]),
    hdrs = glob(
        [
            "include/torch/**/*.h",
            "include/torch/csrc/api/include/**/*.h",
        ],
    ),
    includes = [
        "include",
        "include/torch/csrc/api/include/",
    ],
    deps = [
        ":ATen",
        ":c10_cuda",
    ],
)

cc_library(
    name = "c10_cuda",
    srcs = glob(["lib/libc10_cuda.so"]),
    hdrs = glob([
        "include/c10/**/*.h",
    ]),
    strip_include_prefix = "include",
    deps = [
        ":c10",
    ] + if_cuda_is_configured(
      ["@local_config_cuda//cuda:cuda_headers",
      "@local_config_cuda//cuda:cudart"]
    ) + if_rocm_is_configured(
      ["@local_config_rocm//rocm:rocm_headers",],
    ),
)


cc_library(
    name = "c10",
    srcs = ["lib/libc10.so"],
    hdrs = glob([
        "include/c10/**/*.h",
    ]),
    strip_include_prefix = "include",
)

cc_library(
    name = "ATen",
    hdrs = glob([
        "include/ATen/**/*.h",
    ]),
    strip_include_prefix = "include",
)

cc_library(
    name = "torch_python",
    srcs = [
            "lib/libtorch_python.so",
        ],
    hdrs = glob([
        "include/pybind11/**/*.h",
    ]),
    deps = [
        ":libtorch",
    ],
    strip_include_prefix = "include",
)
