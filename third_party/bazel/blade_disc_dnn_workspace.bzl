load("@org_tensorflow//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("@org_third_party//bazel:common.bzl", "maybe_http_archive")
load("@org_third_party//bazel/onednn:onednn_configure.bzl", "onednn_configure")

def _blade_disc_dnn_repositories():
    # mkldnn cmake external rules
    maybe_http_archive(
        name = "mkl_static",
        build_file = "@org_third_party//bazel/mkldnn:mkl_static.BUILD",
        sha256 = "b0f4f03c5a2090bc1194f348746396183cfb63a5a379d6e86f7fa89006abe28b",
        urls = [
            "https://pai-blade.oss-accelerate.aliyuncs.com/build_deps/mkl/mkl-static-2022.0.1-intel_117.tar.bz2",
            "https://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/build_deps/mkl/mkl-static-2022.0.1-intel_117.tar.bz2",
        ],
    )

    maybe_http_archive(
        name = "mkl_include",
        build_file = "@org_third_party//bazel/mkldnn:mkl_include.BUILD",
        sha256 = "3df729b9fa66f2e1e566c70baa6799b15c9d0e5d3890b9bd084e02299af25002",
        urls = [
            "https://pai-blade.oss-accelerate.aliyuncs.com/build_deps/mkl/mkl-include-2022.0.1-h8d4b97c_803.tar.bz2",
            "https://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/build_deps/mkl/mkl-include-2022.0.1-h8d4b97c_803.tar.bz2",
        ],
    )

    tf_http_archive(
        name = "acl_compute_library",
        sha256 = "c3a443e26539f866969242e690cf0651ef629149741ee18732f954c734da6763",
        strip_prefix = "ComputeLibrary-23.02.1",
        build_file = "@org_third_party//bazel/acl:acl.BUILD",
        patch_file = [
            "@org_third_party//bazel/acl:acl_makefile.patch",
            "@org_third_party//bazel/acl:acl_yitian.patch",
            "@org_third_party//bazel/acl:acl_gemm_hybrid_indirect.patch",
        ],
        urls = tf_mirror_urls("https://github.com/ARM-software/ComputeLibrary/archive/v23.02.1.tar.gz"),
    )

    # ACL_ROOT setting is done in `onednn_configure`
    onednn_configure(name = "local_config_onednn")

def workspace():
    _blade_disc_dnn_repositories()

blade_disc_dnn_workspace = workspace
