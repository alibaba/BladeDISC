load("@org_tensorflow//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")


def workspace():
    tf_http_archive(
        name = "iree-dialects",
        sha256 = "434baa552841885c9f733694966e9ddb0a2eb1844216e546743f84b4655c120a",
        strip_prefix = "iree-candidate-20230613.551/llvm-external-projects/iree-dialects",
        urls = tf_mirror_urls("https://github.com/openxla/iree/archive/refs/tags/candidate-20230613.551.zip"),
        # patch_file = ["@org_disc_compiler//third_party/iree:StructuredTransformOpsExt.cpp.patch"],
    )

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
disc_compiler_workspace = workspace

