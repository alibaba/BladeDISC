diff --git a/BUILD b/BUILD
index 2fb26050..c2744d5b 100644
--- a/BUILD
+++ b/BUILD
@@ -19,7 +19,7 @@ config_setting(
 # ZLIB configuration
 ################################################################################

-ZLIB_DEPS = ["@zlib//:zlib"]
+ZLIB_DEPS = ["@zlib"]

 ################################################################################
 # Protobuf Runtime Library
@@ -218,7 +218,7 @@ cc_library(
 # TODO(keveman): Remove this target once the support gets added to Bazel.
 cc_library(
     name = "protobuf_headers",
-    hdrs = glob(["src/**/*.h"]),
+    hdrs = glob(["src/**/*.h", "src/**/*.inc"]),
     includes = ["src/"],
     visibility = ["//visibility:public"],
 )
