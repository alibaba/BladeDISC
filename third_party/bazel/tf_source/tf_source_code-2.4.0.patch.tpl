diff --git a/tensorflow/stream_executor/BUILD b/tensorflow/stream_executor/BUILD
index d6333d54c1e..4202138adb8 100644
--- a/tensorflow/stream_executor/BUILD
+++ b/tensorflow/stream_executor/BUILD
@@ -12,7 +12,7 @@ load("//tensorflow/core/platform:build_config_root.bzl", "if_static")
 load("//tensorflow/stream_executor:build_defs.bzl", "stream_executor_friends")

 package(
-    default_visibility = [":friends"],
+    default_visibility = ["//visibility:public"],
     licenses = ["notice"],  # Apache 2.0
 )

@@ -155,6 +155,7 @@ cc_library(
         "@com_google_absl//absl/strings",
         "@com_google_absl//absl/synchronization",
     ],
+    alwayslink = 1,
 )

 cc_library(
@@ -412,6 +413,7 @@ cc_library(
         "//tensorflow/stream_executor/lib",
         "//tensorflow/stream_executor/platform",
     ],
+    alwayslink = 1,
 )

 tf_proto_library(
diff --git a/third_party/gpus/cuda_configure.bzl b/third_party/gpus/cuda_configure.bzl
index 3ba34470b93..704003b7f63 100644
--- a/third_party/gpus/cuda_configure.bzl
+++ b/third_party/gpus/cuda_configure.bzl
@@ -551,13 +551,6 @@ def _find_libs(repository_ctx, check_cuda_libs_script, cuda_config):
             cuda_config.cublas_version,
             static = False,
         ),
-        "cublasLt": _check_cuda_lib_params(
-            "cublasLt",
-            cpu_value,
-            cuda_config.config["cublas_library_dir"],
-            cuda_config.cublas_version,
-            static = False,
-        ),
         "cusolver": _check_cuda_lib_params(
             "cusolver",
             cpu_value,
@@ -787,7 +780,6 @@ def _create_dummy_repository(repository_ctx):
             "%{cudart_static_linkopt}": _cudart_static_linkopt(cpu_value),
             "%{cudart_lib}": lib_name("cudart", cpu_value),
             "%{cublas_lib}": lib_name("cublas", cpu_value),
-            "%{cublasLt_lib}": lib_name("cublasLt", cpu_value),
             "%{cusolver_lib}": lib_name("cusolver", cpu_value),
             "%{cudnn_lib}": lib_name("cudnn", cpu_value),
             "%{cufft_lib}": lib_name("cufft", cpu_value),
@@ -819,7 +811,6 @@ filegroup(name="cudnn-include")
         "cuda/cuda/lib/%s" % lib_name("cudart_static", cpu_value),
     )
     repository_ctx.file("cuda/cuda/lib/%s" % lib_name("cublas", cpu_value))
-    repository_ctx.file("cuda/cuda/lib/%s" % lib_name("cublasLt", cpu_value))
     repository_ctx.file("cuda/cuda/lib/%s" % lib_name("cusolver", cpu_value))
     repository_ctx.file("cuda/cuda/lib/%s" % lib_name("cudnn", cpu_value))
     repository_ctx.file("cuda/cuda/lib/%s" % lib_name("curand", cpu_value))
@@ -1011,13 +1002,11 @@ def _create_local_cuda_repository(repository_ctx):
             cublas_include_path + "/cublas.h",
             cublas_include_path + "/cublas_v2.h",
             cublas_include_path + "/cublas_api.h",
-            cublas_include_path + "/cublasLt.h",
         ],
         outs = [
             "cublas/include/cublas.h",
             "cublas/include/cublas_v2.h",
             "cublas/include/cublas_api.h",
-            "cublas/include/cublasLt.h",
         ],
     ))

@@ -1158,7 +1147,6 @@ def _create_local_cuda_repository(repository_ctx):
             "%{cudart_static_linkopt}": _cudart_static_linkopt(cuda_config.cpu_value),
             "%{cudart_lib}": _basename(repository_ctx, cuda_libs["cudart"]),
             "%{cublas_lib}": _basename(repository_ctx, cuda_libs["cublas"]),
-            "%{cublasLt_lib}": _basename(repository_ctx, cuda_libs["cublasLt"]),
             "%{cusolver_lib}": _basename(repository_ctx, cuda_libs["cusolver"]),
             "%{cudnn_lib}": _basename(repository_ctx, cuda_libs["cudnn"]),
             "%{cufft_lib}": _basename(repository_ctx, cuda_libs["cufft"]),
diff --git a/third_party/gpus/rocm/BUILD.tpl b/third_party/gpus/rocm/BUILD.tpl
index ecbb4b5cebc..eb427c23d6b 100644
--- a/third_party/gpus/rocm/BUILD.tpl
+++ b/third_party/gpus/rocm/BUILD.tpl
@@ -11,6 +11,13 @@ config_setting(
     },
 )

+config_setting(
+    name = "using_dcu",
+    values = {
+        "define": "using_dcu=true",
+    },
+)
+
 cc_library(
     name = "rocm_headers",
     hdrs = [
diff --git a/third_party/gpus/rocm/build_defs.bzl.tpl b/third_party/gpus/rocm/build_defs.bzl.tpl
index ce4c1b04399..775afc8d3e3 100644
--- a/third_party/gpus/rocm/build_defs.bzl.tpl
+++ b/third_party/gpus/rocm/build_defs.bzl.tpl
@@ -47,3 +47,15 @@ def if_rocm_is_configured(x):
     if rocm_is_configured():
       return x
     return []
+
+def if_dcu(if_true, if_false = []):
+    """Shorthand for select()'ing on whether we're building with DCU.
+
+    Returns a select statement which evaluates to if_true if we're building
+    with DCU enabled.  Otherwise, the select statement evaluates to if_false.
+
+    """
+    return select({
+        "@local_config_rocm//rocm:using_dcu": if_true,
+        "//conditions:default": if_false
+    })
