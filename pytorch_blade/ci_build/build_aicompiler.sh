set -ex
function help {
    echo "Usage: $0 --py_ver PYTHON_VERSION [--ral_cxx11_abi] [--cpu_only] [--dcu]"
}

function require {
    value=$1
    flag=$2
    if [[ "${value}" == "" ]]; then
        echo "Required flag is missing: ${flag}"
        help
        exit -1;
    fi
}

ARGS=`getopt -o h --long py_ver:,ral_cxx11_abi,dcu,cpu_only -n "$0" -- "$@"`
if [ $? != 0 ]; then
    echo "Error when preparing arguments..."
    exit 1
fi

eval set -- "${ARGS}"

PYTHON_VERSION=""
AICOMPILER_CXX11_ABI_CONFIG=""
AICOMPILER_CPU_ONLY_CONFIG=""
AICOMPILER_DCU_CONFIG=""

while true
do
    case "$1" in
        --py_ver)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --ral_cxx11_abi)
            AICOMPILER_CXX11_ABI_CONFIG="--ral_cxx11_abi"
            shift
            ;;
        --dcu)
            AICOMPILER_DCU_CONFIG="--dcu"
            shift
            ;;
        --cpu_only)
            AICOMPILER_CPU_ONLY_CONFIG="--cpu_only"
            shift
            ;;
        -h)
            help
            exit
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

require "${PYTHON_VERSION}" "--py_ver"

echo "PYTHON_VERSION=${PYTHON_VERSION}"
echo "AICOMPILER_CXX11_ABI_CONFIG=${AICOMPILER_CXX11_ABI_CONFIG}"
echo "AICOMPILER_CPU_ONLY_CONFIG=${AICOMPILER_CPU_ONLY_CONFIG}"
echo "AICOMPILER_DCU_CONFIG=${AICOMPILER_DCU_CONFIG}"
echo "TORCH_BLADE_PLATFORM_ALIBABA=${TORCH_BLADE_PLATFORM_ALIBABA}"

if [ "${TORCH_BLADE_PLATFORM_ALIBABA}" == "ON" ]; then
CI_SCRIPTS_DIR=./platform_alibaba/ci_build/
else
CI_SCRIPTS_DIR=./scripts/python/
fi

# Currently TVM is not needed and would break build_mlir_ral
export TF_NEED_TVM=0
AICOMPILER_VENV_DIR=$(dirname $(dirname $(which ${PYTHON_VERSION})))
${CI_SCRIPTS_DIR}/tao_build.py ${AICOMPILER_VENV_DIR} --compiler-gcc default --bridge-gcc default --stage configure_pytorch ${AICOMPILER_CXX11_ABI_CONFIG} ${AICOMPILER_CPU_ONLY_CONFIG} ${AICOMPILER_DCU_CONFIG}
${CI_SCRIPTS_DIR}/tao_build.py ${AICOMPILER_VENV_DIR} --stage build_mlir_ral ${AICOMPILER_CXX11_ABI_CONFIG} ${AICOMPILER_CPU_ONLY_CONFIG} ${AICOMPILER_DCU_CONFIG}

