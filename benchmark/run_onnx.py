#!/usr/bin/env python3

import timeit
import numpy as np
import onnxruntime as rt

sess_options = rt.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
sess = rt.InferenceSession("model.onnx", sess_options)

def run():
    sess.run(["output"], {"input": np.random.rand(10000, 3).astype(np.float32)})

times = timeit.repeat("run()", number=100, repeat=5, globals=globals())
print(f"{np.mean(times):.3f}+/-{np.std(times):.3f}s")
