# MKL

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

%{mkl_static_lib_imports}

cc_library(
    name = "mkl_headers",
    hdrs = glob(["include/*.h"]),
    includes = ["include"],
)

cc_library(
    name = "mkl_static",
    linkstatic = 1,
    alwayslink = 1,
    deps = [
        ":mkl_headers",
%{mkl_static_lib_targets}
    ],
)

exports_files(["%{mkl_iomp_dynamic_lib_target}"])
