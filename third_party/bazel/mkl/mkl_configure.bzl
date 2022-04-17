"""Setup TensorFlow as external dependency"""

_MKL_ROOT = "MKL_ROOT"
_WITH_MKL = "BLADE_WITH_MKL"

_MKL_STATIC_LIBS = [
    "mkl/lib/intel64_lin/libmkl_intel_ilp64.a",
    "mkl/lib/intel64_lin/libmkl_intel_thread.a",
    "mkl/lib/intel64_lin/libmkl_core.a",
    "compiler/lib/intel64_lin/libiomp5.a",
]

_MKL_IOMP5_SO = "compiler/lib/intel64_lin/libiomp5.so"

def _tpl(repository_ctx, tpl, substitutions):
    repository_ctx.template(
        tpl,
        Label("//bazel/mkl:%s.tpl" % tpl),
        substitutions,
    )

def _cc_import(repository_ctx, lib_path):
    fname = repository_ctx.path(lib_path).basename
    return """
cc_import(
    name = "{}",
    static_library = "{}",
    alwayslink = 0,
)""".format(fname, lib_path)

def _mkl_configure_impl(repository_ctx):
    with_mkl = repository_ctx.os.environ.get(_WITH_MKL, "0").lower()
    if with_mkl not in ["1", "true", "on"]:
        _tpl(repository_ctx, "BUILD", {
            "%{mkl_static_lib_targets}": "",
            "%{mkl_static_lib_imports}": "",
        })
        _tpl(repository_ctx, "build_defs.bzl", {
            "%{if_mkl}": "if_false",
        })
        return

    root = repository_ctx.os.environ[_MKL_ROOT].rstrip("/")
    header_dir = root + "/mkl/include"

    repository_ctx.symlink(header_dir, "include")
    repository_ctx.symlink(root + "/", "lib")

    static_lib_imports = [
        _cc_import(repository_ctx, "lib/" + lib)
        for lib in _MKL_STATIC_LIBS
    ]
    static_lib_targets = "\n".join([
        '        ":' + repository_ctx.path(lib).basename + '",'
        for lib in _MKL_STATIC_LIBS
    ])
    dynamic_lib_target = "lib/" + _MKL_IOMP5_SO

    _tpl(repository_ctx, "BUILD", {
        "%{mkl_static_lib_targets}": static_lib_targets,
        "%{mkl_static_lib_imports}": "\n\n".join(static_lib_imports),
        "%{mkl_iomp_dynamic_lib_target}": dynamic_lib_target,
    })
    _tpl(repository_ctx, "build_defs.bzl", {
        "%{if_mkl}": "if_true",
    })

mkl_configure = repository_rule(
    implementation = _mkl_configure_impl,
    environ = [
        _MKL_ROOT,
        _WITH_MKL,
    ],
)
