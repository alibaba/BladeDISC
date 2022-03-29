def python_bin_path():
    return "%{PYTHON_BIN_PATH}"

def cc_bin_path():
    return "%{CC_BIN_PATH}"

def cxx_bin_path():
    return "%{CXX_BIN_PATH}"

def nvcc_bin_path():
    return "%{NVCC_BIN_PATH}"

def cuda_home():
    return "%{CUDA_HOME}"

def cuda_version():
    return "%{CUDA_VERSION}"

def if_hie_enabled(if_true, if_false = []):
    if %{IF_HIE}:
        return if_true
    return if_false

def if_hie_disabled(if_true, if_false = []):
    if not %{IF_HIE}:
        return if_true
    return if_false

def if_tensorrt_enabled(if_true, if_false = []):
    if %{TENSORRT_ENABLED}:
        return if_true
    return if_false

def if_tensorrt_disabled(if_true, if_false = []):
    if not %{TENSORRT_ENABLED}:
        return if_true
    return if_false

def if_internal(if_true, if_false = []):
    if %{IF_INTERNAL}:
        return if_true
    return if_false
