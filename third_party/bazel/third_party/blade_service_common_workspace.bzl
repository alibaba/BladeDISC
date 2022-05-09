load("@org_third_party//bazel:common.bzl", "maybe_http_archive")

# Current workspace is for platform_alibaba build with USE_BLADE_SERVICE_COMMON
def _blade_service_common_repositories():
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

def workspace():
    _blade_service_common_repositories()

blade_service_common_workspace = workspace
