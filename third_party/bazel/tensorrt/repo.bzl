load("//bazel:common.bzl", "files_exist")

_TENSORRT_INSTALL_PATH = "TENSORRT_INSTALL_PATH"

def _cc_import_myelin():
    return """
cc_import(
    name = "myelin_compiler_static",
    static_library = "lib/libmyelin_compiler_static.a",
)

cc_import(
    name = "myelin_executor_static",
    static_library = "lib/libmyelin_executor_static.a",
)

cc_import(
    name = "myelin_pattern_library_static",
    static_library = "lib/libmyelin_pattern_library_static.a",
)

cc_import(
    name = "myelin_pattern_runtime_static",
    static_library = "lib/libmyelin_pattern_runtime_static.a",
)

cc_library(
    name = "myelin_static",
    deps = [
        ":myelin_compiler_static",
        ":myelin_executor_static",
        ":myelin_pattern_library_static",
        ":myelin_pattern_runtime_static",
    ]
)

"""

def warn(msg):
    print("{red}{msg}{nc}".format(red = "\033[0;31m", msg = msg, nc = "\033[0m"))

def _impl(repo_ctx):
    tensorrt_path = repo_ctx.os.environ.get(_TENSORRT_INSTALL_PATH, None)
    if tensorrt_path == None:
        warn("Please set the customize tensorrt library path via env var: {}".format(_TENSORRT_INSTALL_PATH))
        tensorrt_path = "/usr/local/TensorRT/"

    repo_ctx.symlink(tensorrt_path + "/include", "include")
    repo_ctx.symlink(tensorrt_path + "/lib", "lib")

    if_has_myelin = all(files_exist(
        repo_ctx,
        [
            "lib/libmyelin_compiler_static.a",
            "lib/libmyelin_executor_static.a",
            "lib/libmyelin_pattern_library_static.a",
            "lib/libmyelin_pattern_runtime_static.a",
        ],
    ))

    repo_ctx.template("BUILD", Label("//bazel/tensorrt:trt.BUILD.tpl"), {
        "%{myelin_static_rule}": _cc_import_myelin() if if_has_myelin else "",
    })
    repo_ctx.template("build_defs.bzl", Label("//bazel/tensorrt:build_defs.bzl.tpl"), {
        "%{IF_HAS_MYELIN}": "True" if if_has_myelin else "False",
    })

tensorrt_configure = repository_rule(
    implementation = _impl,
    local = True,
    environ = [_TENSORRT_INSTALL_PATH],
)
