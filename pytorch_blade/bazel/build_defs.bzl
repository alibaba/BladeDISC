
def if_tensorrt_enabled(if_true, if_false = []):
    return select({
        "//:enable_tensorrt": if_true,
        "//conditions:default": if_false,
    })
