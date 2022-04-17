package(default_visibility = ["//visibility:public"])

cc_import(
    name = "cudart_static",
    static_library = "lib64/libcudart_static.a",
)

cc_import(
    name = "cublas_static",
    static_library = "lib64/libcublas_static.a",
)

%{import_cublaslt}

cc_import(
    name = "culibos_static",
    static_library = "lib64/libculibos_static.a",
)

cc_import(
    name = "cudnn_static",
    static_library = "lib64/libcudnn_static.a",
)

cc_import(
    name = "nvrtc",
    shared_library = "lib64/libnvrtc.so",
)
