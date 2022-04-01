def if_has_cublaslt(if_true, if_false = []):
    if %{IF_HAS_CUBLASLT}:
        return if_true
    return if_false
