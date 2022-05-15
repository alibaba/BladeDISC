
cc_library(
    name = "xla_builder",
    srcs = ["xla_builder.cc"],
    hdrs = ["xla_builder.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@org_tensorflow//tensorflow/compiler/xla/client:xla_builder_deps",
    ],
)

cc_library(
    name = "type_to_shape",
    srcs = [
       "type_to_shape.cc",
       "shape_util.cc", ],
    hdrs = ["type_to_shape.h", "shape_util.h", ],
    visibility = ["//visibility:public"],
    deps = [
        "@org_tensorflow//tensorflow/compiler/mlir/xla:type_to_shape_deps",
    ],
)

cc_library(
    name = "mhlo_builder",
    srcs = ["mlir_hlo_builder.cc", "shape_inference.cc", "mlir_shape_builder.cpp"],
    hdrs = ["mlir_hlo_builder.h", "shape_inference.h", "mlir_shape_builder.h", "macros.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":type_to_shape",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:mlir_hlo_builder_deps",
    ],
)

