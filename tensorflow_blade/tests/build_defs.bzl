load("@rules_python//python:defs.bzl", "py_test")

def tf_blade_ops_py_tests(srcs, deps = [], data = [], tags = None):
    """Create a py_test for each .py file in srcs."""
    all_deps = deps + ["//tests/custom_ops:tf_blade_ops_ut_common"]
    for file in srcs:
        if not file.endswith(".py"):
            fail("Need .py file in srcs, but got: " + file)
        name = file.rsplit(".", 1)[0]
        py_test(
            name = name,
            srcs = [file],
            deps = all_deps,
            data = data,
            tags = tags,
        )
