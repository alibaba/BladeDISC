# Copyright 2021 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import logging

import torch_blade._torch_blade._tools as tools

# create logger
logger = logging.getLogger(__name__)
# Default logging nothing
if tools.read_bool_from_env('TORCH_BLADE_DEBUG_LOG', False):
    logger.addHandler(logging.NullHandler())

@contextlib.contextmanager
def logger_level_context(level):
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    # create formatter
    formatter = logging.Formatter('%(asctime)s %(filename)s:%(lineno)d - [%(levelname)-05s] %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    old_level = logger.level
    try:
        logger.addHandler(ch)
        logger.setLevel(level)
        yield
    finally:
        logger.setLevel(old_level)
        logger.removeHandler(ch)
