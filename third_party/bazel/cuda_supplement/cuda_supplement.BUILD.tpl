package(default_visibility = ["//visibility:public"])

cc_import(
    name = "cudart_static",
    static_library = "lib64/libcudart_static.a",
)

cc_import(
    name = "cublas_static",
    static_library = "lib64/libcublas_static.a",
)

cc_import(
    name = "cublasLt_static",
    static_library = "lib64/libcublasLt_static.a",
)

cc_import(
    name = "culibos_static",
    static_library = "lib64/libculibos.a",
)

cc_import(
    name = "cudnn_static",
    static_library = "libcudnn_static_patched.a",
)

cc_import(
    name = "cudnn_adv_infer_static",
    static_library = "lib64/libcudnn_adv_infer_static.a",
)

cc_import(
    name = "cudnn_cnn_infer_static",
    static_library = "lib64/libcudnn_cnn_infer_static.a",
)

cc_import(
    name = "cudnn_cnn_train_static",
    static_library = "lib64/libcudnn_cnn_train_static.a",
)

cc_import(
    name = "cudnn_ops_infer_static",
    static_library = "lib64/libcudnn_ops_infer_static.a",
)

cc_import(
    name = "cudnn_ops_train_static",
    static_library = "lib64/libcudnn_ops_train_static.a",
)

cc_import(
    name = "nvrtc",
    shared_library = "lib64/libnvrtc.so",
)
