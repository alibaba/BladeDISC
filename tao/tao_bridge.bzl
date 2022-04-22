def tao_bridge_cc_test(
        name,
        srcs,
        deps = [],
        data = [],
        linkstatic = 0,
        copts = [],
        linkopts = [],
        **kwargs):
    native.cc_test(
        name = name,
        srcs = srcs,
        copts = copts,
        linkopts = [
            "-lpthread",
            "-lm",
            "-ldl",
        ] + linkopts,
        deps = deps + [
            ":tao_bridge",
            "@com_google_googletest//:gtest_main",
            "@local_config_tf//:tf_header_lib",
            "@local_config_tf//:libtensorflow_framework",
        ],
        data = data,
        linkstatic = select({
            "@local_config_blade_disc_helper//:is_mkldnn": 1,
            "//conditions:default": linkstatic,
        }),
        **kwargs
    )
