package(default_visibility = ["//visibility:public"])

config_setting(
    name = "aarch64_linux",
    constraint_values = [
        "@platforms//cpu:aarch64",
        "@platforms//os:linux",
    ],
)

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

cc_library(
    name = "nvinfer_headers",
    hdrs = select({
        ":aarch64_linux": [
            "include/aarch64-linux-gnu/NvUtils.h",
        ] + glob(
            [
                "include/aarch64-linux-gnu/NvInfer*.h",
            ],
            exclude = [
                "include/aarch64-linux-gnu/NvInferPlugin.h",
                "include/aarch64-linux-gnu/NvInferPluginUtils.h",
            ],
        ),
        ":windows": [
            "include/NvUtils.h",
        ] + glob(
            [
                "include/NvInfer*.h",
            ],
            exclude = [
                "include/NvInferPlugin.h",
                "include/NvInferPluginUtils.h",
            ],
        ),
        "//conditions:default": [
            "include/NvUtils.h",
        ] + glob(
            [
                "include/NvInfer*.h",
            ],
            exclude = [
                "include/NvInferPlugin.h",
                "include/NvInferPluginUtils.h",
            ],
        ),
    }),
    includes = select({
        ":aarch64_linux": ["include/aarch64-linux-gnu"],
        ":windows": ["include/"],
        "//conditions:default": ["include/"],
    }),
    visibility = ["//visibility:private"],
)

cc_import(
    name = "nvinfer_static_lib",
    static_library = select({
        ":aarch64_linux": "lib/aarch64-linux-gnu/libnvinfer_static.a",
        ":windows": "lib/nvinfer.lib",
        "//conditions:default": "lib/libnvinfer_static.a",
    }),
    visibility = ["//visibility:private"],
)

cc_import(
    name = "nvinfer_lib",
    shared_library = select({
        ":aarch64_linux": "lib/aarch64-linux-gnu/libnvinfer.so",
        ":windows": "lib/nvinfer.dll",
        "//conditions:default": "lib/libnvinfer.so",
    }),
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvinfer",
    visibility = ["//visibility:public"],
    deps = [
        "nvinfer_headers",
        "nvinfer_lib",
        "@local_config_cuda//cuda:cudart",
        #"@cudnn",
    ] + select({
        ":windows": ["@local_config_cuda//cuda:cublas"],
        "//conditions:default": ["@local_config_cuda//cuda:cublas"],
    }),
)

####################################################################################

cc_import(
    name = "nvparsers_lib",
    shared_library = select({
        ":aarch64_linux": "lib/aarch64-linux-gnu/libnvparsers.so",
        ":windows": "lib/nvparsers.dll",
        "//conditions:default": "lib/libnvparsers.so",
    }),
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvparsers_headers",
    hdrs = select({
        ":aarch64_linux": [
            "include/aarch64-linux-gnu/NvCaffeParser.h",
            "include/aarch64-linux-gnu/NvOnnxParser.h",
            "include/aarch64-linux-gnu/NvOnnxParserRuntime.h",
            "include/aarch64-linux-gnu/NvOnnxConfig.h",
            "include/aarch64-linux-gnu/NvUffParser.h",
        ],
        ":windows": [
            "include/NvCaffeParser.h",
            "include/NvOnnxParser.h",
            "include/NvOnnxParserRuntime.h",
            "include/NvOnnxConfig.h",
            "include/NvUffParser.h",
        ],
        "//conditions:default": [
            "include/NvCaffeParser.h",
            "include/NvOnnxParser.h",
            "include/NvOnnxConfig.h",
            "include/NvUffParser.h",
        ] + glob([
            "include/NvOnnxParserRuntime.h"]),
    }),
    includes = select({
        ":aarch64_linux": ["include/aarch64-linux-gnu"],
        ":windows": ["include/"],
        "//conditions:default": ["include/"],
    }),
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvparsers",
    visibility = ["//visibility:public"],
    deps = [
        "nvinfer",
        "nvparsers_headers",
        "nvparsers_lib",
    ],
)

####################################################################################

cc_import(
    name = "nvonnxparser_lib",
    shared_library = select({
        ":aarch64_linux": "lib/aarch64-linux-gnu/libnvonnxparser.so",
        ":windows": "lib/nvonnxparser.dll",
        "//conditions:default": "lib/libnvonnxparser.so",
    }),
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvonnxparser_headers",
    hdrs = select({
        ":aarch64_linux": [
            "include/aarch64-linux-gnu/NvOnnxParser.h",
            "include/aarch64-linux-gnu/NvOnnxParserRuntime.h",
            "include/aarch64-linux-gnu/NvOnnxConfig.h",
        ],
        ":windows": [
            "include/NvOnnxParser.h",
            "include/NvOnnxParserRuntime.h",
            "include/NvOnnxConfig.h",
        ],
        "//conditions:default": [
            "include/NvOnnxParser.h",
            "include/NvOnnxConfig.h",
        ] + glob([
            "include/NvOnnxParserRuntime.h",]),
    }),
    includes = select({
        ":aarch64_linux": ["include/aarch64-linux-gnu"],
        ":windows": ["include/"],
        "//conditions:default": ["include/"],
    }),
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvonnxparser",
    visibility = ["//visibility:public"],
    deps = [
        "nvinfer",
        "nvonnxparser_headers",
        "nvonnxparser_lib",
    ],
)

####################################################################################

cc_import(
    name = "nvonnxparser_runtime_lib",
    shared_library = select({
        ":aarch64_linux": "lib/libnvonnxparser_runtime.so",
        ":windows": "lib/nvonnxparser_runtime.dll",
        "//conditions:default": "lib/libnvonnxparser_runtime.so",
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nvonnxparser_runtime_header",
    hdrs = select({
        ":aarch64_linux": [
            "include/aarch64-linux-gnu/NvOnnxParserRuntime.h",
        ],
        ":windows": [
            "include/NvOnnxParserRuntime.h",
        ],
        "//conditions:default": [
            "include/NvOnnxParserRuntime.h",
        ],
    }),
    includes = select({
        ":aarch64_linux": ["include/aarch64-linux-gnu"],
        ":windows": ["include/"],
        "//conditions:default": ["include/"],
    }),
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvonnxparser_runtime",
    visibility = ["//visibility:public"],
    deps = [
        "nvinfer",
        "nvparsers_headers",
        "nvparsers_lib",
    ],
)

####################################################################################

cc_import(
    name = "nvcaffeparser_lib",
    shared_library = select({
        ":aarch64_linux": "lib/aarch64-linux-gnu/libnvcaffe_parsers.so",
        ":windows": "lib/nvcaffe_parsers.dll",
        "//conditions:default": "lib/libnvcaffe_parsers.so",
    }),
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvcaffeparser_headers",
    hdrs = select({
        ":aarch64_linux": [
            "include/aarch64-linux-gnu/NvCaffeParser.h",
        ],
        ":windows": [
            "include/NvCaffeParser.h",
        ],
        "//conditions:default": [
            "include/NvCaffeParser.h",
        ],
    }),
    includes = select({
        ":aarch64_linux": ["include/aarch64-linux-gnu"],
        ":windows": ["include/"],
        "//conditions:default": ["include/"],
    }),
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvcaffeparser",
    deps = [
        "nvcaffeparser_headers",
        "nvcaffeparser_lib",
        "nvinfer",
    ],
    visibility = ["//visibility:public"],
)

####################################################################################

cc_library(
    name = "nvinferplugin",
    hdrs = select({
        ":aarch64_linux": glob(["include/aarch64-linux-gnu/NvInferPlugin*.h"]),
        ":windows": glob(["include/NvInferPlugin*.h"]),
        "//conditions:default": glob(["include/NvInferPlugin*.h"]),
    }),
    srcs = select({
        ":aarch64_linux": ["lib/aarch64-linux-gnu/libnvinfer_plugin.so"],
        ":windows": ["lib/nvinfer_plugin.dll"],
        "//conditions:default": ["lib/libnvinfer_plugin.so"],
    }),
    includes = select({
        ":aarch64_linux": ["include/aarch64-linux-gnu/"],
        ":windows": ["include/"],
        "//conditions:default": ["include/"],
    }),
    deps = [
        "nvinfer",
        "@local_config_cuda//cuda:cudart",
        # "@cudnn",
    ] + select({
        ":windows": ["@local_config_cuda//cuda:cublas"],
        "//conditions:default": ["@local_config_cuda//cuda:cublas"],
    }),
    alwayslink = True,
    copts = [
        "-pthread"
    ],
    linkopts = [
        "-lpthread",
    ] + select({
        ":aarch64_linux": ["-Wl,--no-as-needed -ldl -lrt -Wl,--as-needed"],
        "//conditions:default": []
    })
)
