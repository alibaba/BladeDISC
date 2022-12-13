# Copyright 2022 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# type: ignore
import logging
import sys
import os

PY_VER = "{}.{}".format(sys.version_info.major, sys.version_info.minor)

ENV_VAR_TMP_GCC = "BLADE_TMP_GCC"


def __create_logger():
    """Create a logger with color."""
    # The background is set with 40 plus the number of the color, and the foreground with 30
    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

    # These are the sequences need to get colored ouput
    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[1;%dm"
    BOLD_SEQ = "\033[1m"

    COLORS = {
        "WARNING": YELLOW,
        "INFO": GREEN,
        "DEBUG": BLUE,
        "CRITICAL": YELLOW,
        "ERROR": RED,
    }

    class ColoredFormatter(logging.Formatter):
        def __init__(self, msg, use_color=False):
            logging.Formatter.__init__(self, msg)
            self.use_color = use_color

        def format(self, record):
            levelname = record.levelname
            if self.use_color and levelname in COLORS:
                levelname_color = (
                    COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
                )
                record.levelname = levelname_color
            return logging.Formatter.format(self, record)

    class ColoredLogger(logging.Logger):
        FORMAT = "{}%(asctime)s{} %(levelname)19s %(message)s".format(
            BOLD_SEQ, RESET_SEQ
        )

        def __init__(self, name):
            logging.Logger.__init__(self, name, logging.DEBUG)
            color_formatter = ColoredFormatter(
                self.FORMAT, use_color=sys.stdout.isatty() and sys.stderr.isatty()
            )
            console = logging.StreamHandler()
            console.setFormatter(color_formatter)
            self.addHandler(console)
            return

    logging.setLoggerClass(ColoredLogger)
    logger = logging.getLogger("blade_ci")
    logger.setLevel(logging.INFO)
    return logger


logger = __create_logger()



def get_trt_version(trt_home):
    hdr = os.path.join(trt_home, "include", "NvInferVersion.h")
    with open(hdr, "r") as f:
        major, minor, patch = None, None, None
        for line in f.readlines():
            line = line.strip()
            if "#define NV_TENSORRT_SONAME_MAJOR" in line:
                major = line.split(" ")[2]
            elif "#define NV_TENSORRT_SONAME_MINOR" in line:
                minor = line.split(" ")[2]
            elif "#define NV_TENSORRT_SONAME_PATCH" in line:
                patch = line.split(" ")[2]
        if None in [major, minor, patch]:
            raise Exception(f"Failed to decuce TensorRT version from: {hdr}")
        return ".".join([major, minor, patch])
