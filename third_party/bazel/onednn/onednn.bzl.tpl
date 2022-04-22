def _filter_static_lib_impl(ctx):
    return DefaultInfo(files=depset([f for f in ctx.files.srcs if f.basename.endswith(".a")]))

filter_static_lib = rule(
    implementation = _filter_static_lib_impl,
    attrs = {
        "srcs": attr.label_list(),
    },
)
