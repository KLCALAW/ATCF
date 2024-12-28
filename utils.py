import sys

import numpy as np
from scipy.stats import norm


# Progress bar
#-------------
def progress_bar(progress):
    bar_length = 50
    block = int(round(bar_length * progress))
    text = "\rProgress: [{0}] {1:.0f}%    ".format('█' * block + "-" * (bar_length - block), progress * 100)
    sys.stdout.write(text)
    sys.stdout.flush()

def clear_progress_bar():
    sys.stdout.write("\r")
    sys.stdout.flush()
    sys.stdout.write(" " * 100 + "\r")
    sys.stdout.flush()