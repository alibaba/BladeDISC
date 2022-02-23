
The following symlink file was introduced to resolve bazel build error:
`java -> ../../../../tf_community/tensorflow/tools/toolchains/java/`

The detail error message:
```
ERROR: /root/.cache/bazel/_bazel_root/604a0c21dbd7bf6513f5a00fab4c110a/external/bazel_tools/tools/jdk/BUILD:72:28: every rule of type java_toolchain_alias implicitly depends upon the target '//tensorflow/tools/toolchains/java:tf_java_toolchain', but this target could not be found be
cause of: no such package 'tensorflow/tools/toolchains/java': BUILD file not found in any of the following directories. Add a BUILD file to a directory to mark it as a package.
```
