import os
import contextlib
import logging

# create logger
logger = logging.getLogger(__name__)
# Default logging nothing
if os.environ.get('TORCH_ADDONS_DEBUG_LOG', None) is None:
    logger.addHandler(logging.NullHandler())

@contextlib.contextmanager
def logger_level_context(level):
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    # create formatter
    formatter = logging.Formatter('%(asctime)s - [%(levelname)-05s] %(message)s')
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
