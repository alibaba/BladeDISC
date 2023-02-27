load("@org_tensorflow//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")


def workspace():
    tf_http_archive(
        name = "iree-dialects",
        sha256 = "f6ed70146b85d27c25cb4653cfb879d7954be426516b1cac230cbf5bb8b041da",
        strip_prefix = "iree-7cd0a8cb6e1027188faa75b9361c4b10aab4707c/llvm-external-projects/iree-dialects",
        urls = tf_mirror_urls("https://github.com/iree-org/iree/archive/7cd0a8cb6e1027188faa75b9361c4b10aab4707c.zip"),
        patch_file = ["@org_disc_compiler//third_party/iree:StructuredTransformOpsExt.cpp.patch"],
    )

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
disc_compiler_workspace = workspace

