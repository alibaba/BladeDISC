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

def _find_recursive(repository_ctx, dir):
    result = repository_ctx.execute(["find", "-L", dir, "-type", "f"])
    if result.return_code != 0:
        fail("Repository command failed: \n %s" % result.stderr.strip())
    return result.stdout.strip().splitlines()

def _cc_import(repository_ctx, lib_path):
    fname = repository_ctx.path(lib_path).basename
    return """
cc_import(
    name = "{}",
    static_library = "{}",
    alwayslink = 0,
)""".format(fname, lib_path)

def _genrule_copy(repository_ctx, name, srcs, outs):
    """Returns a rule to copy a set of files."""
    cmds = []

    # Copy files.
    for src, out in zip(srcs, outs):
        cmds.append('cp -f "%s" "$(RULEDIR)/%s"' % (src, out))
    outs = [('        "%s",' % out) for out in outs]
    return """genrule(
    name = "%s",
    tags = ["no-remote-cache"],
    outs = [
%s
    ],
    cmd = \"""%s \""",
)""" % (name, "\n".join(outs), " && \\\n             ".join(cmds))

def _mkl_configure_impl(repository_ctx):
    with_mkl = repository_ctx.os.environ.get(_WITH_MKL, "0").lower()
    if with_mkl not in ["1", "true", "on"]:
        _tpl(repository_ctx, "BUILD", {
            "%{mkl_copy_rules}": "",
            "%{mkl_static_lib_targets}": "",
            "%{mkl_static_lib_imports}": "",
        })
        _tpl(repository_ctx, "build_defs.bzl", {
            "%{if_mkl}": "if_false",
        })
        return

    root = repository_ctx.os.environ[_MKL_ROOT].rstrip("/")
    headers = _find_recursive(repository_ctx, root + "/mkl/include")

    copy_rules = [
        _genrule_copy(
            repository_ctx,
            name = "copy_mkl_include",
            srcs = headers,
            outs = ["include/" + repository_ctx.path(header).basename for header in headers],
        ),
        _genrule_copy(
            repository_ctx,
            name = "copy_mkl_static_libs",
            srcs = [root + "/" + lib for lib in _MKL_STATIC_LIBS],
            outs = ["lib/" + lib for lib in _MKL_STATIC_LIBS],
        ),
        _genrule_copy(
            repository_ctx,
            name = "copy_iomp5",
            srcs = [root + "/" + _MKL_IOMP5_SO],
            outs = ["lib/" + _MKL_IOMP5_SO],
        ),
    ]

    static_lib_imports = [
        _cc_import(repository_ctx, "lib/" + lib)
        for lib in _MKL_STATIC_LIBS
    ]
    static_lib_targets = "\n".join([
        '        ":' + repository_ctx.path(lib).basename + '",'
        for lib in _MKL_STATIC_LIBS
    ])

    _tpl(repository_ctx, "BUILD", {
        "%{mkl_copy_rules}": "\n\n".join(copy_rules),
        "%{mkl_static_lib_targets}": static_lib_targets,
        "%{mkl_static_lib_imports}": "\n\n".join(static_lib_imports),
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
