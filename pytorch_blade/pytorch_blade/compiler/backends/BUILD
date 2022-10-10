package(default_visibility = ["//visibility:public"])

cc_library(
    name = "torch_blade_backends",
    srcs = [
        "engine_class.cpp",
        "engine_interface.cpp",
        "backend_input_outputs.cpp",
    ],
    hdrs = [
        "engine_class.h",
        "engine_interface.h",
        "backend_input_outputs.h",
    ],
    deps = [
        "//pytorch_blade/common_utils:torch_blade_common",
        "//pytorch_blade/compiler/jit:torch_blade_jit",
    ],
    alwayslink = 1,
)

