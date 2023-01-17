load(
  "//tensorflow:tensorflow.bzl",
  "tf_cc_binary",
  "tf_cc_test",
)

load("//tensorflow/compiler/mlir/disc:disc.bzl",
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
        "@org_disc//ral:ral_base_context_lib",
        "//tensorflow/compiler/mlir/disc:all_passes",
        "//tensorflow/compiler/decoupling:tao_compiler_input",
        "//tensorflow/compiler/decoupling:tao_compiler",
        "//tensorflow/compiler/xla/mlir_hlo:all_passes",
        "//tensorflow/core:lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core/platform:subprocess",
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
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_googletest//:gtest",
    ],
    data = [
        "//tensorflow/compiler/mlir/disc/tools/disc-replay/test_data:data.tar",
        "//tensorflow/compiler/mlir/disc/tools/disc-replay/test_data:program.pb"
    ]
)
