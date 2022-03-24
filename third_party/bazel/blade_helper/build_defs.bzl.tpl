def python_bin_path():
    return "%{PYTHON_BIN_PATH}"

def if_tensorrt_enabled(x):
    if %{TENSORRT_ENABLED}:
        return select({"//conditions:default": x})
    return select({"//conditions:default": []})

def if_tensorrt_disabled(x):
    if %{TENSORRT_ENABLED}:
        return select({"//conditions:default": []})
    return select({"//conditions:default": x})

def if_internal(x):
    if %{IF_INTERNAL}:
        return select({"//conditions:default": x})
    return select({"//conditions:default": []})
