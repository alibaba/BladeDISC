
def if_tensorrt_enabled(if_true, if_false = []):
    return select({
        "//:enable_tensorrt": if_true,
        "//conditions:default": if_false,
    })

def if_platform_alibaba(if_true, if_false = []):
    return select({
        "//:platform_alibaba": if_true,
        "//conditions:default": if_false,
    })
