def if_pai_tf(x):
    if %{IS_PAI_TF}:
        return select({"//conditions:default": x})
    return select({"//conditions:default": []})

def if_not_pai_tf(x):
    if %{IS_PAI_TF}:
        return select({"//conditions:default": []})
    return select({"//conditions:default": x})

def tf_copts():
    return %{TF_COPTS}

def tf_lib_dir():
    return "%{TF_LIB_DIR}"

def if_tf2(x):
    if %{IS_TF2}:
        return select({"//conditions:default": x})
    return select({"//conditions:default": []})

def tf_version():
    return "%{TF_VERSION}"

def tf_major_version():
    return "%{TF_MAJOR_VERSION}"
