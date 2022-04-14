#!/bin/bash

function log {
    now=`date '+%Y-%m-%d %H:%M:%S.%3N'`
    if [ -t 1 ]; then
        echo -e "${now} \033[32m\e[1m [INFO]\e[0m $@"
    else
        echo -e "${now} [INFO] $@"
    fi
}

function usage {
    echo "$0 --os linux --arch arm64-v8a|arn64-v9 [--build_neon]"
}

set -e

ARGS=`getopt -o h --long build_neon,os:,arch: -n "$0" -- "$@"`
if [ $? != 0 ]; then
    usage
    exit 1
fi
eval set -- "${ARGS}"

os="linux"
arch=""
build_neon=0

while true; do
    case "$1" in
        --os)
            os=$2
            shift 2
            ;;
        --arch)
            arch=$2
            shift 2
            ;;
        --build_neon)
            build_neon=1
            shift 1
            ;;
        -h)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Internal error!"
            exit 1
            ;;
    esac
done

pushd external/acl_compute_library

log "=================== Build Arguments ==================="
log "Current dir      : $(pwd)"
log "os               : ${os}"
log "arch             : ${arch}"
log "build_neon       : ${build_neon}"

scons --silent -j$(($(nproc) - 1)) Werror=0 debug=0 neon=${build_neon} opencl=0 embed_kernels=0 os=${os} arch=${arch} build=native extra_cxx_flags="-fPIC"

# a workaround for static linking
rm -f build/libarm*.so
mv build/libarm_compute-static.a build/libarm_compute.a
mv build/libarm_compute_core-static.a build/libarm_compute_core.a
mv build/libarm_compute_graph-static.a build/libarm_compute_graph.a

# BUILD.bazel is linked from dir where target acl_compute_library should be
# origin_path=`readlink -f BUILD.bazel`
# origin_dir=$(dirname $origin_path)
# echo $origin_dir
# cp -r build/ $origin_dir
popd
