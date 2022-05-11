load("//bazel:common.bzl", "auto_config_fail", "files_exist", "get_bash_bin")

_TENSORRT_INSTALL_PATH = "TENSORRT_INSTALL_PATH"

def _cc_import_myelin():
    return """
cc_import(
    name = "myelin_compiler_static",
    static_library = "lib/libmyelin_compiler_static.a",
)

cc_import(
    name = "myelin_executor_static",
    static_library = "lib/libmyelin_executor_static.a",  # use patched one
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
    if if_has_myelin:
        # Since tensorflow enables --whole-archive for all static libs(see 
        # https://github.com/tensorflow/tensorflow/blob/master/.bazelrc#L130 ), which
        # causes symbol conflicts when myelin static libraries are linked. More
        # specifically, libmyelin_compiler_static.a and libmyelin_executor_static.a have
        # common symbols, `multiple definition` error will be raised when both of them
        # are linked.
        # To work around this issue, we weaken symbols in libmyelin_executor_static.a
        # which also appears in libmyelin_compiler_static.a. This can be removed once
        # tensorflow doesn't depend on Bazel for whole-archive linking or we drop
        # supporting for lower version TensorRT with myelin libraries.
        patch_myelin_sh = repo_ctx.path(Label("//bazel/tensorrt:patch_myelin.sh"))
        result = repo_ctx.execute([get_bash_bin(repo_ctx), patch_myelin_sh, tensorrt_path, "./"])
        if result.return_code != 0:
            auto_config_fail("Failed to patch myelin static libraries.\nstdout: {}\nstderr: {}".format(
                result.stdout,
                result.stderr,
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
