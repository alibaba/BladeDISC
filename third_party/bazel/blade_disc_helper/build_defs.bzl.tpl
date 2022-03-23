def python_bin_path():
    return "%{PYTHON_BIN_PATH}"

def if_tensorrt_enabled(x):
    if %{TENSORRT_ENABLED}:
        return select({"//conditions:default": x})
    return select({"//conditions:default": []})

def if_internal(x):
    if %{IF_INTERNAL}:
        return select({"//conditions:default": x})
    return select({"//conditions:default": []})

def tao_build_version():
    return "%{TAO_BUILD_VERSION}"

def tao_build_git_branch():
    return "%{TAO_BUILD_GIT_BRANCH}"

def tao_build_git_head():
    return "%{TAO_BUILD_GIT_HEAD}"

def tao_build_host():
    return "%{TAO_BUILD_HOST}"

def tao_build_ip():
    return "%{TAO_BUILD_IP}"

def tao_build_time():
    return "%{TAO_BUILD_TIME}"
