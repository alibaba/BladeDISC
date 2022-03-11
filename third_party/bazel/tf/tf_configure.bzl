"""Setup TensorFlow as external dependency"""

_BLADE_WITH_TF = "BLADE_WITH_TF"
_TF_MAJOR_VERSION = "TF_MAJOR_VERSION"
_TF_MINOR_VERSION = "TF_MINOR_VERSION"
_TF_IS_PAI = "TF_IS_PAI"
_TF_HEADER_DIR = "TF_HEADER_DIR"
_TF_SHARED_LIBRARY_DIR = "TF_SHARED_LIBRARY_DIR"
_TF_SHARED_LIBRARY_NAME = "TF_SHARED_LIBRARY_NAME"

_TF_HEADER_LIB = "cc_library(\n \
   name = \"tf_header_lib\",\n \
   hdrs = [\":tf_header_include\"],\n \
   includes = [\"include\"],\n \
   visibility = [\"//visibility:public\"],\n \
)\n \
"
_TF_LIB = "cc_library(\n \
   name = \"libtensorflow_framework\",\n \
   srcs = [\":tensorflow_framework_lib_file\"],\n \
   linkstatic=1,\n \
   visibility = [\"//visibility:public\"],\n \
)\n \
"

def _tpl(repository_ctx, tpl, substitutions):
    repository_ctx.template(
        tpl,
        Label("//bazel/tf:%s.tpl" % tpl),
        substitutions,
    )

def _fail(msg):
    """Output failure message when auto configuration fails."""
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("%sConfigure Error:%s %s\n" % (red, no_color, msg))

def _genrule(name, outs, cmd):
    """Returns a string that is a genrule."""
    return """genrule(
    name = "{}",
    outs = {},
    cmd = \"\"\"
{}
\"\"\",
)""".format(name, outs, cmd)

def _execute(
        repository_ctx,
        cmdline,
        error_msg = None,
        empty_stdout_fine = False):
    result = repository_ctx.execute(cmdline)
    if result.return_code != 0:
        _fail("Repository command failed: \n %s" % result.stderr.strip())
    return result

def _find_recursive(repository_ctx, dir):
    result = _execute(repository_ctx, ["find", "-L", dir, "-type", "f"])
    return result.stdout.strip().splitlines()

def _copy_and_genrule(
        repository_ctx,
        src_dir,
        dst_dir,
        genrule_name,
        src_files = []):
    """Create bazel genrule that copy files to bazel's internal directory."""
    src_dir = src_dir.rstrip("/")
    dst_dir = dst_dir.rstrip("/")
    if src_files:
        src_items = [src_dir + "/" + f for f in src_files]
        one_shot_cp_command = ''
    else:
        src_items = _find_recursive(repository_ctx, src_dir)
        one_shot_cp_command = 'cp -r {}/* "$(RULEDIR)/{}"'.format(src_dir, dst_dir)
    commands = []
    outs = []
    for item in src_items:
        item_path = item.replace(src_dir + "/", "")
        out = dst_dir + "/" + item_path
        cmd = 'cp -f "{}" "$(RULEDIR)/{}"'.format(item, out)
        outs.append(out)
        commands.append(cmd)
    if one_shot_cp_command:
        return _genrule(genrule_name, str(outs), one_shot_cp_command)
    else:
        return _genrule(genrule_name, str(outs), " && ".join(commands))

def _tf_configure_impl(repository_ctx):
    with_tf = repository_ctx.os.environ[_BLADE_WITH_TF].lower()
    if with_tf not in ["1", "true", "on"]:
        _tpl(repository_ctx, "BUILD", {
            "%{TF_HEADER_LIB}": "",
            "%{TF_LIB}": "",
            "%{TF_HEADER_GENRULE}": "",
            "%{TF_LIB_GENRULE}": "",
        })
        _tpl(repository_ctx, "build_defs.bzl", {
            "%{IS_PAI_TF}": "True",
            "%{TF_COPTS}": "[]",
            "%{TF_LIB_DIR}": "",
            "%{IS_TF2}": "False",
        })
        return

    tf_header_dir = repository_ctx.os.environ[_TF_HEADER_DIR]
    tf_header_genrule = _copy_and_genrule(repository_ctx, tf_header_dir, "include", "tf_header_include")
    tf_lib_dir = repository_ctx.os.environ[_TF_SHARED_LIBRARY_DIR]
    tf_lib_name = repository_ctx.os.environ[_TF_SHARED_LIBRARY_NAME]
    tf_lib_genrule = _copy_and_genrule(repository_ctx, tf_lib_dir, "lib", "tensorflow_framework_lib_file", src_files = [tf_lib_name])

    _tpl(repository_ctx, "BUILD", {
        "%{TF_HEADER_LIB}": _TF_HEADER_LIB,
        "%{TF_LIB}": _TF_LIB,
        "%{TF_HEADER_GENRULE}": tf_header_genrule,
        "%{TF_LIB_GENRULE}": tf_lib_genrule,
    })

    tf_major = repository_ctx.os.environ[_TF_MAJOR_VERSION]
    tf_minor = repository_ctx.os.environ[_TF_MINOR_VERSION]
    tf_is_pai = repository_ctx.os.environ[_TF_IS_PAI].lower() in ["1", "true", "on"]
    tf_copt = [
        "-DTF_{}_{}".format(tf_major, tf_minor),
        "-DTF_MAJOR={}".format(tf_major),
        "-DTF_MINOR={}".format(tf_minor),
    ]
    if tf_is_pai:
        tf_copt.append("-DTF_IS_PAI")
    _tpl(repository_ctx, "build_defs.bzl", {
        "%{IS_PAI_TF}": "True" if tf_is_pai else "False",
        "%{TF_COPTS}": "[\"" + "\", \"".join(tf_copt) + "\"]",
        "%{TF_LIB_DIR}": tf_lib_dir,
        "%{IS_TF2}": "True" if tf_major == "2" else "False",
    })

tf_configure = repository_rule(
    implementation = _tf_configure_impl,
    environ = [
        _BLADE_WITH_TF,
        _TF_HEADER_DIR,
        _TF_SHARED_LIBRARY_DIR,
        _TF_SHARED_LIBRARY_NAME,
    ],
)
