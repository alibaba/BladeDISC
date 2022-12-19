
def if_tensorrt_enabled(if_true, if_false = []):
    return select({
        "//:enable_tensorrt": if_true,
        "//conditions:default": if_false,
    })

def if_tensorrt_static_linked(if_true, if_false = []):
    return select({
        "//:enable_static_trt": if_true,
        "//conditions:default": if_false,
    })

def if_platform_alibaba(if_true, if_false = []):
    return select({
        "//:platform_alibaba": if_true,
        "//conditions:default": if_false,
    })

def if_ltc_disc_backend(if_true, if_false=[]):
    return select({
        "//:enable_ltc_disc_backend": if_true,
        "//conditions:default": if_false,
    })

def if_quantization_enabled(if_true, if_false=[]):
    return select({
        "//:enable_quantization": if_true,
        "//conditions:default": if_false,
    })

def if_neural_engine_enabled(if_true, if_false=[]):
    return select({
        "//:enable_neural_engine": if_true,
        "//conditions:default": if_false,
    })
