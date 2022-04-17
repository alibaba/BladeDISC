# Build configurations for MKL.

def if_mkl_enabled(if_true, if_false=[]):
  """Tests whether MKL was enabled during the configure process."""
  return %{if_mkl}

def mkl_copts():
    return if_mkl_enabled([
        "-DUSE_AVX512",
        "-DMKL_ILP64",
        "-mfma",
        "-mavx",
        "-mavx2",
        "-mavx512f",
        "-fopenmp",
        "-m64",
    ])
