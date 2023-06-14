load("@org_tensorflow//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")


def workspace():
    tf_http_archive(
        name = "iree-dialects",
        sha256 = "c6e73a659835b919f1e80529501063dc4d9e1868124caf7fdfbe3dc6049c038b",
        strip_prefix = "iree-8abd1bdb35b2a90790a3751bee2c3f435bd2521e/llvm-external-projects/iree-dialects",
        urls = tf_mirror_urls("https://github.com/iree-org/iree/archive/8abd1bdb35b2a90790a3751bee2c3f435bd2521e.zip"),
        patch_file = ["@org_disc_compiler//third_party/iree:StructuredTransformOpsExt.cpp.patch"],
    )

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
disc_compiler_workspace = workspace

