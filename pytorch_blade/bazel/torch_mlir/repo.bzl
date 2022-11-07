"""Utilities for defining Blade Bazel dependencies."""

def _get_link_dict(ctx, link_files, build_file):
    link_dict = {ctx.path(v): ctx.path(Label(k)) for k, v in link_files.items()}
    if build_file:
        # Use BUILD.bazel because it takes precedence over BUILD.
        link_dict[ctx.path("BUILD.bazel")] = ctx.path(Label(build_file))
    return link_dict

def _blade_http_archive_impl(ctx):
    # Construct all paths early on to prevent rule restart. We want the
    # attributes to be strings instead of labels because they refer to files
    # in the TensorFlow repository, not files in repos depending on TensorFlow.
    # See also https://github.com/bazelbuild/bazel/issues/10515.
    link_dict = _get_link_dict(ctx, ctx.attr.link_files, ctx.attr.build_file)

    # For some reason, we need to "resolve" labels once before the
    # download_and_extract otherwise it'll invalidate and re-download the
    # archive each time.
    # https://github.com/bazelbuild/bazel/issues/10515
    patch_files = ctx.attr.patch_file
    for patch_file in patch_files:
        if patch_file:
            ctx.path(Label(patch_file))

    ctx.download_and_extract(
        url = ctx.attr.urls,
        sha256 = ctx.attr.sha256,
        type = ctx.attr.type,
        stripPrefix = ctx.attr.strip_prefix,
    )
    if patch_files:
        for patch_file in patch_files:
            patch_file = ctx.path(Label(patch_file)) if patch_file else None
            if patch_file:
                ctx.patch(patch_file, strip = 1)

    for dst, src in link_dict.items():
        ctx.delete(dst)
        ctx.symlink(src, dst)

_blade_http_archive = repository_rule(
    implementation = _blade_http_archive_impl,
    attrs = {
        "sha256": attr.string(mandatory = True),
        "urls": attr.string_list(mandatory = True),
        "strip_prefix": attr.string(),
        "type": attr.string(),
        "patch_file": attr.string_list(),
        "build_file": attr.string(),
        "system_build_file": attr.string(),
        "link_files": attr.string_dict(),
        "system_link_files": attr.string_dict(),
    },
)

def blade_http_archive(name, sha256, urls, **kwargs):
    if native.existing_rule(name):
        print("\n\033[1;33mWarning:\033[0m skipping import of repository '" +
              name + "' because it already exists.\n")
        return

    _blade_http_archive(
        name = name,
        sha256 = sha256,
        urls = urls,
        **kwargs
    )
