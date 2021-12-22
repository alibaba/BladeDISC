# -*- coding: utf-8 -*-
import os
import re
import sys
from deepmd.entrypoints.main import main
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import blade_disc_tf as disc


if __name__ == '__main__':
    # Enable DISC.
    disc.enable()

    # Run with DeePMD-kit API.
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(main())
