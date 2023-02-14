load(
  "@org_tensorflow//tensorflow:tensorflow.bzl",
  "tf_cc_binary",
  "tf_cc_test",
)

load("//mlir/util:util.bzl",
     "disc_cc_library",
     "if_cuda_or_rocm",
)

disc_cc_library(
    name = "disc-replay",
    srcs = [
        "record.cc",
        "disc_interpreter.cc"
    ],
    hdrs = [
        "record.h",
        "tar_helper.h",
        "disc_interpreter.h"
    ],
    deps = [
        "//mlir/disc:all_passes",
        "//decoupling:tao_compiler_input",
        "//decoupling:tao_compiler",
        "//mlir/xla/ral:ral_base_context_lib",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:all_passes",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core/platform:subprocess",
    ] + if_cuda_or_rocm([
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cuda_driver",
        "@local_config_cuda//cuda:cudart",
    ])
)

tf_cc_binary(
    name = "disc-replay-main",
    srcs = [
        "disc_replay_main.cc"
    ],
    deps = [
        ":disc-replay"
    ]
)

tf_cc_test(
    name = "disc-replay-test",
    srcs = [
        "replay_test.cc"
    ],
    deps = [
        ":disc-replay",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:test_main",
        "@com_google_googletest//:gtest",
    ],
    data = [
        "//mlir/disc/tools/disc-replay/test_data:data.tar",
        "//mlir/disc/tools/disc-replay/test_data:program.pb"
    ]
)
