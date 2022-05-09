# Import repository rules.
load("@org_third_party//bazel/blade_disc_helper:blade_disc_helper_configure.bzl", "blade_disc_helper_configure")
load("@org_third_party//bazel:common.bzl", "maybe_http_archive")
load("@org_third_party//bazel/onednn:onednn_configure.bzl", "onednn_configure")
load("@org_third_party//bazel/tf:tf_configure.bzl", "tf_configure")
load("@org_third_party//bazel/tf_protobuf:tf_protobuf_configure.bzl", "tf_protobuf_configure")

load("@org_tensorflow//third_party/gpus:cuda_configure.bzl", "cuda_configure")
load("@org_tensorflow//third_party/gpus:rocm_configure.bzl", "rocm_configure")
load("@org_tensorflow//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

# Import external repository rules.
load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def _tao_bridge_repositories():
    maybe_http_archive(
        name = "bazel_skylib",
        sha256 = "1c531376ac7e5a180e0237938a2536de0c54d93f5c278634818e0efc952dd56c",
        urls = [
            "http://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/build_deps/bazel-skylib-1.0.3.tar.gz",
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/releases/download/1.0.3/bazel-skylib-1.0.3.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.3/bazel-skylib-1.0.3.tar.gz",
        ],
    )

    maybe_http_archive(
        name = "cub_archive",
        build_file = "@org_third_party//bazel/third_party:cub.BUILD",
        sha256 = "162514b3cc264ac89d91898b58450190b8192e2af1142cf8ccac2d59aa160dda",
        strip_prefix = "cub-1.9.9",
        urls = [
            "http://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/build_deps/cub_archive/1.9.9.zip",
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/NVlabs/cub/archive/1.9.9.zip",
            "https://github.com/NVlabs/cub/archive/1.9.9.zip",
        ],
    )

    rules_foreign_cc_dependencies(
        register_built_tools = False,  # do not build cmake/make from source,
        register_default_tools = False,  # nor download from official site,
        register_preinstalled_tools = True,  # just use the pre-installed.
    )

    # work around for rules_jave download failures
    maybe_http_archive(
        name = "rules_java",
        sha256 = "f5a3e477e579231fca27bf202bb0e8fbe4fc6339d63b38ccb87c2760b533d1c3",
        strip_prefix = "rules_java-981f06c3d2bd10225e85209904090eb7b5fb26bd",
        urls = ["http://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/bazelbuild/rules_java/archive/981f06c3d2bd10225e85209904090eb7b5fb26bd.tar.gz"],
    )

    # mkldnn cmake external rules
    maybe_http_archive(
        name = "mkl_static",
        build_file = "@org_third_party//bazel/mkldnn:mkl_static.BUILD",
        sha256 = "b0f4f03c5a2090bc1194f348746396183cfb63a5a379d6e86f7fa89006abe28b",
        urls = [
            "https://hlomodule.oss-cn-zhangjiakou.aliyuncs.com/mkl_package/mkl-static-2022.0.1-intel_117.tar.bz2",
            "https://hlomodule.oss-cn-zhangjiakou.aliyuncs.com/mkl_package/mkl-static-2022.0.1-intel_117.tar.bz2",
        ],
    )

    maybe_http_archive(
        name = "mkl_include",
        build_file = "@org_third_party//bazel/mkldnn:mkl_include.BUILD",
        sha256 = "3df729b9fa66f2e1e566c70baa6799b15c9d0e5d3890b9bd084e02299af25002",
        urls = [
            "https://hlomodule.oss-cn-zhangjiakou.aliyuncs.com/mkl_package/mkl-include-2022.0.1-h8d4b97c_803.tar.bz2",
            "https://hlomodule.oss-cn-zhangjiakou.aliyuncs.com/mkl_package/mkl-include-2022.0.1-h8d4b97c_803.tar.bz2",
        ],
    )

    tf_http_archive(
        name = "acl_compute_library",
        sha256 = "11244b05259fb1c4af7384d0c3391aeaddec8aac144774207582db4842726540",
        strip_prefix = "ComputeLibrary-22.02",
        build_file = "@org_third_party//bazel/acl:acl.BUILD",
        patch_file = ["@org_third_party//bazel/acl:acl_makefile.patch"],
        urls = tf_mirror_urls("https://github.com/ARM-software/ComputeLibrary/archive/v22.02.tar.gz"),
    )

    # ACL_ROOT setting is done in `onednn_configure`
    onednn_configure(name = "local_config_onednn")
    native.new_local_repository(
        name = "onednn",
        build_file = "@local_config_onednn//:onednn.BUILD",
        path = "third_party/mkldnn"
    )

    maybe_http_archive(
        name = "libuuid",
        build_file = "@org_third_party//bazel/third_party:libuuid.BUILD",
        sha256 = "46af3275291091009ad7f1b899de3d0cea0252737550e7919d17237997db5644",
        strip_prefix = "libuuid-1.0.3",
        urls = [
            "http://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/build_deps/libuuid/libuuid-1.0.3.tar.gz",
            "https://udomain.dl.sourceforge.net/project/libuuid/libuuid-1.0.3.tar.gz",
        ],
    )

    maybe_http_archive(
        name = "openssl",
        build_file = "@org_third_party//bazel/third_party:openssl.BUILD",
        sha256 = "892a0875b9872acd04a9fde79b1f943075d5ea162415de3047c327df33fbaee5",
        strip_prefix = "openssl-1.1.1k",
        urls = [
            "http://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/build_deps/openssl/openssl-1.1.1k.tar.gz",
            "https://www.openssl.org/source/openssl-1.1.1k.tar.gz",
        ],
    )

    native.new_local_repository(
        name = "blade_gemm",
        build_file = "@org_third_party//bazel/third_party:blade_gemm.BUILD",
        path = "../../platform_alibaba/blade_gemm"
    )

    # These are for platform_alibaba build with USE_BLADE_SERVICE_COMMON
    maybe_http_archive(
        name = "zlib",
        build_file = "@org_third_party//bazel/third_party:zlib.BUILD",
        sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
        strip_prefix = "zlib-1.2.11",
        urls = [
            "http://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/build_deps/zlib/zlib-1.2.11.tar.gz",
            "https://storage.googleapis.com/mirror.tensorflow.org/zlib.net/zlib-1.2.11.tar.gz",
            "https://zlib.net/zlib-1.2.11.tar.gz",
        ],
    )

    maybe_http_archive(
        name = "platform_alibaba_curl",
        build_file = "@org_third_party//bazel/third_party:curl.BUILD",
        sha256 = "01ae0c123dee45b01bbaef94c0bc00ed2aec89cb2ee0fd598e0d302a6b5e0a98",
        strip_prefix = "curl-7.69.1",
        urls = [
            "http://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/build_deps/curl/curl-7.69.1.tar.gz",
            "https://curl.haxx.se/download/curl-7.69.1.tar.gz",
        ],
    )

    maybe_http_archive(
        name = "aliyun_log_sdk",
        build_file = "@org_third_party//bazel/third_party:aliyun_log_sdk.BUILD",
        sha256 = "06780bcb128d9082fb671534552bf7ce859253f36f8c7aa13603dbe0ce2acfd7",
        strip_prefix = "aliyun-log-c-sdk-blade_20201130-d42054fbadcf558ddb4ccde98307af09b544dac7",
        urls = [
            "http://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/build_deps/aliyun-log-c-sdk/aliyun-log-c-sdk-blade_20201130-d42054fbadcf558ddb4ccde98307af09b544dac7.tar.gz",
        ],
    )

    # Only the following symbols has conflicts now for tao_bridge
    # mbedtls_md5_clone
    # mbedtls_md5_starts
    # mbedtls_md5_process
    # mbedtls_md5_update
    # mbedtls_md5_finish
    # mbedtls_md5_init
    # mbedtls_md5
    maybe_http_archive(
        name = "apes",
        build_file = "@org_third_party//bazel/third_party:apes.BUILD",
        patch_cmds = [
            "echo -e 'mbedtls_md5_clone\nmbedtls_md5_starts\nmbedtls_md5_process\nmbedtls_md5_update\nmbedtls_md5_finish\nmbedtls_md5_init\nmbedtls_md5' >> local_sym",
            "objcopy --localize-symbols=local_sym libapes.a",
        ],
        sha256 = "adbd740b4c0d73935783fb1430072169c5c88f430ebf34eaa78b384bf7b47af2",
        strip_prefix = "apes_linux_x86_64_20200302",
        urls = [
            "http://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/build_deps/apes/apes_linux_x86_64_20200302.tgz",
        ],
    )

    maybe_http_archive(
        name = "nlohmann_json",
        build_file = "@org_third_party//bazel/third_party:nlohmann_json.BUILD",
        sha256 = "0ba8ecbfd0406ffb39513f70fc8efcda35b1c35342bbe4635b5df20ea562db62",
        strip_prefix = "json-3.6.1-stripped",
        urls = [
            "http://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/build_deps/nlohmann_json_lib/v3.6.1-stripped.tar.gz",
        ],
    )

def _tao_bridge_toolchains():
    tf_configure(name = "local_config_tf")

    tf_protobuf_configure(name = "local_config_tf_protobuf")

    blade_disc_helper_configure(name = "local_config_blade_disc_helper")

    cuda_configure(name = "local_config_cuda")

    rocm_configure(name = "local_config_rocm")

def workspace():
    _tao_bridge_repositories()
    _tao_bridge_toolchains()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
tao_bridge_workspace0 = workspace
