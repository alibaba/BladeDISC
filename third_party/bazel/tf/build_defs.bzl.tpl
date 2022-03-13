def tf_copts():
    return %{TF_COPTS}

def tf_lib_dir():
    return "%{TF_LIB_DIR}"

def if_tf2(x):
    if %{IS_TF2}:
        return select({"//conditions:default": x})
    return select({"//conditions:default": []})

def tf_version_define():
    return "#define TF_VERSION %{TF_VERSION}"
