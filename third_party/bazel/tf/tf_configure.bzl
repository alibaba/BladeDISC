"""Setup TensorFlow as external dependency"""

_TF_MAJOR_VERSION = "TF_MAJOR_VERSION"
_TF_MINOR_VERSION = "TF_MINOR_VERSION"
_TF_HEADER_DIR = "TF_HEADER_DIR"
_TF_SHARED_LIBRARY_DIR = "TF_SHARED_LIBRARY_DIR"
_TF_SHARED_LIBRARY_NAME = "TF_SHARED_LIBRARY_NAME"

def _tpl(repository_ctx, tpl, substitutions):
    repository_ctx.template(
        tpl,
        Label("//bazel/tf:%s.tpl" % tpl),
        substitutions,
    )

def _tf_configure_impl(repository_ctx):
    tf_header_dir = repository_ctx.os.environ[_TF_HEADER_DIR]
    tf_lib_dir = repository_ctx.os.environ[_TF_SHARED_LIBRARY_DIR]

    repository_ctx.symlink(tf_header_dir, "include")
    repository_ctx.symlink(tf_lib_dir, "lib")
    repository_ctx.symlink(Label("//bazel/tf:BUILD.tpl"), "BUILD")

    tf_major = repository_ctx.os.environ[_TF_MAJOR_VERSION]
    tf_minor = repository_ctx.os.environ[_TF_MINOR_VERSION]
    tf_copt = [
        "-DTF_{}_{}".format(tf_major, tf_minor),
        "-DTF_MAJOR={}".format(tf_major),
        "-DTF_MINOR={}".format(tf_minor),
    ]
    _tpl(repository_ctx, "build_defs.bzl", {
        "%{TF_COPTS}": "[\"" + "\", \"".join(tf_copt) + "\"]",
        "%{TF_LIB_DIR}": tf_lib_dir,
        "%{IS_TF2}": "True" if tf_major == "2" else "False",
        "%{TF_VERSION}": "{}.{}".format(tf_major, tf_minor)
    })

tf_configure = repository_rule(
    implementation = _tf_configure_impl,
    environ = [
        _TF_MAJOR_VERSION,
        _TF_MINOR_VERSION,
        _TF_HEADER_DIR,
        _TF_SHARED_LIBRARY_DIR,
        _TF_SHARED_LIBRARY_NAME,
    ],
)
