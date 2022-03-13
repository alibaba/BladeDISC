def blade_cc_test(
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
        deps = deps + ["@googltest//:gtest_main"],
        data = data,
        linkstatic = linkstatic,
        **kwargs
    )

def if_gpu(if_true, if_false = []):
    return select({
        "//:gpu": if_true,
        "//conditions:default": if_false,
    })

def if_cpu(if_true, if_false = []):
    return select({
        "//:cpu": if_true,
        "//conditions:default": if_false,
    })

def device_name():
    # It's pretty trick to return a list, but take a look at this:
    # https://github.com/bazelbuild/bazel/issues/6643
    return select({
        "//:gpu": ["gpu"],
        "//:cpu": ["cpu"],
    })

def if_tf_supported(if_true, if_false = []):
    return select({
        "//:tf_supported": if_true,
        "//conditions:default": if_false,
    })

def if_tf_unsupported(if_true, if_false = []):
    return select({
        "//:tf_supported": if_false,
        "//conditions:default": if_true,
    })

def if_torch_supported(if_true, if_false = []):
    return select({
        "//:torch_supported": if_true,
        "//conditions:default": if_false,
    })

def if_torch_unsupported(if_true, if_false = []):
    return select({
        "//:torch_supported": if_false,
        "//conditions:default": if_true,
    })
