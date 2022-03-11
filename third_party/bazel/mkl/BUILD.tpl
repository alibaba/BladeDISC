# MKL

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

%{mkl_copy_rules}

%{mkl_static_lib_imports}


cc_library(
    name = "mkl_headers",
    hdrs = [":copy_mkl_include"],
    strip_include_prefix = "include",
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