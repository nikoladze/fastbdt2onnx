#!/usr/bin/env python3

import timeit
import numpy as np
from PyFastBDT import FastBDT

clf = FastBDT.Classifier()
clf.load("FastBDTv5.txt")

def run():
    clf.predict(np.random.rand(10000, 3))

times = timeit.repeat("run()", number=100, repeat=5, globals=globals())
print(f"{np.mean(times):.3f}+/-{np.std(times):.3f}s")
