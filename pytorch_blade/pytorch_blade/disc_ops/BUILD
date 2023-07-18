package(default_visibility = ["//visibility:public"])

cc_library(
    name = "attention_op",
    srcs = [
        "attention_op.cpp",
    ],
    deps = [
        "@local_org_torch//:libtorch",
        "//pytorch_blade/common_utils:torch_blade_common"
    ],
    alwayslink = True,
)
