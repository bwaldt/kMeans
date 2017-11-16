import gzip
import pandas as pd
import numpy as np


def readInFile():
    with gzip.open('R6/ydata-fp-td-clicks-v1_0.20090501.gz', 'rb') as f:
        data = f.read()

if __name__ == "__main__":
   readInFile()
