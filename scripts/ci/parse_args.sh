#!/bin/bash

function help() {
  echo "disc.sh --cpu-only --venv /opt/venv_py3_tf115"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu-only)
      CPU_ONLY="--cpu_only"
      shift
      ;;
    --venv)
      DISC_VENV="$2"
      shift 2
      ;;
    -h)
      help
      exit
      ;;
    *)
      echo "empty"
      shift
      ;;
  esac
done
