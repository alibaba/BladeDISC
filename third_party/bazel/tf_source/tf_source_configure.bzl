load("//bazel:common.bzl", "get_env_bool_value")

# tf serving's versions are the same with it's corresponding tensorflow version
# e.g. tf-serving 2.4.1 uses tensorflow==2.4.1's source code
_TF_SERVING_VERSION = "TF_SERVING_VERSION"

tf_serving_version_dict = {
    "2.4.0": [
        "9c94bfec7214853750c7cacebd079348046f246ec0174d01cd36eda375117628",  # sha256
        "582c8d236cb079023657287c318ff26adb239002",  # git_commit
    ],
    "2.4.1": [
        "ac2d19cf529f9c2c9faaf87e472d08a2bdbb2ab058958e2cafd65e5eb0637b2b",  # sha256
        "85c8b2a817f95a3e979ecd1ed95bff1dc1335cff",  # git_commit
    ],
    "2.7.0": [
        "ff0df77ec72676d3260502dd19f34518ecd65bb9ead4f7dfdf8bd11cff8640e3",  # sha256
        "c256c071bb26e1e13b4666d1b3e229e110bc914a",  # git_commit
    ],
}

def _tf_source_configure_impl(repository_ctx):
    if _TF_SERVING_VERSION not in repository_ctx.os.environ:
        repository_ctx.template("BUILD", Label("//bazel/tf_source:BUILD.tpl"), {})
        repository_ctx.template("tf_source_workspace1.bzl", Label("//bazel/tf_source:no_tf_serving_workspace1.bzl.tpl"), {})
        repository_ctx.template("tf_source_workspace0.bzl", Label("//bazel/tf_source:no_tf_serving_workspace0.bzl.tpl"), {})
    else:
        tf_serving_version = repository_ctx.os.environ[_TF_SERVING_VERSION]
        tf_serving_info = tf_serving_version_dict.get(tf_serving_version)
        if tf_serving_info == None:
            fail("The tf_serving version - {} is not supported now!".format(tf_serving_version))
        repository_ctx.template(
            "tf_source_workspace1.bzl",
            Label("//bazel/tf_source:tf_source_workspace1.bzl.tpl"),
            {
                "%{TF_SOURCE_SHA256}": tf_serving_info[0],
                "%{TF_SOURCE_GIT_COMMIT}": tf_serving_info[1],
            },
        )
        repository_ctx.template("tf_source_workspace0.bzl", Label("//bazel/tf_source:tf_source_workspace0.bzl.tpl"), {})
        repository_ctx.template("BUILD", Label("//bazel/tf_source:BUILD.tpl"), {})
        repository_ctx.template("tf_source_code.patch", Label("//bazel/tf_source:tf_source_code-{}.patch.tpl".format(tf_serving_version)), {})

tf_source_configure = repository_rule(
    implementation = _tf_source_configure_impl,
    environ = [
        _TF_SERVING_VERSION,
    ],
)
