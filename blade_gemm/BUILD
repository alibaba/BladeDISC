licenses(["notice"])  # Apache 2.0

# Link the built blade_gemm lib to bazel
# Notice: this need to be built before called by bazel
# You just need to:
#
#     mkdir build
#     cd build
#     cmake .. -DCUTLASS_NVCC_ARCHS="80" -DCUTLASS_LIBRARY_KERNELS=s1688tf32gemm,f16_s1688gemm_f16,f16_s16816gemm_f16,s16816tf32gemm -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_UNITY_BUILD_ENABLED=ON
#     make -j
#

cc_library(
    name = "blade_gemm",
    hdrs = glob(["include/*"]),
    srcs = [
        "build/libblade_gemm.a",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
)
