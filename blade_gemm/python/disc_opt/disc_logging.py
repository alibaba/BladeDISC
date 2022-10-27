import logging
import logging.handlers
import sys

logger = logging.getLogger('DISC')


rf_handler = logging.StreamHandler(sys.stderr)
rf_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

logger.addHandler(rf_handler)
logger.setLevel(logging.INFO)



