package(default_visibility = ["//visibility:public"])

load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "if_cuda_is_configured",
)

cc_library(
    name = "libtorch",
    srcs = [
            "lib/libtorch.so",
            "lib/libtorch_cpu.so",
            "lib/libtorch_global_deps.so",
        ] + glob(
            ["lib/libtorch_cuda.so","lib/libtensorpipe.so",]),
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

cc_library(
    name = "ltc_ts_backend",
    srcs = glob(
        [
            "lazy_tensor_core/lazy_tensor_core/**/*.cpp",
            "ts_include/torch/csrc/lazy/python/*.cpp",
            "lazy_tensor_core/third_party/computation_client/*.cc",
        ],
        exclude=[
            "lazy_tensor_core/test/**/*",
            "lazy_tensor_core/**/init_python_bindings.cpp",
            "ts_include/torch/csrc/lazy/**/test_*.cpp"
        ],
        allow_empty=False
    ),
    hdrs = glob(
        [
            "lazy_tensor_core/lazy_tensor_core/csrc/ts_backend/*.h",
            "lazy_tensor_core/third_party/computation_client/*.h",
            "ts_include/torch/csrc/lazy/**/*.h",
        ],
        allow_empty=False
    ),
    includes =[
        "pytorch/torch/include",
        "pytorch/torch/csrc",
        "lazy_tensor_core",
        "include",
        "include/torch/csrc/api/include/",
        "ts_include"
    ],
    deps = [
         "@local_config_python//:python_headers",
    ],
    visibility = ["//visibility:public"],
)
