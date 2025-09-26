import numpy as np
import onnxruntime as ort
import pytest
from PyFastBDT.FastBDT import Classifier

from fastbdt2onnx import convert
from fastbdt2onnx.bdt import BDT


def test_consistent_fastbdt_onnx():
    rng = np.random.default_rng(42)
    with open("data/FastBDTv5.txt") as f:
        bdt = BDT.from_file(f)
    model = convert("data/FastBDTv5.txt")
    splittings = [
        [
            cut.index
            for tree in bdt.forest.trees
            for cut in tree.cuts
            if cut.feature == i
        ]
        for i in range(bdt.numberOfFeatures)
    ]
    starts = [np.nanmin(s).item() - 0.1 for s in splittings]
    stops = [np.nanmax(s).item() + 0.1 for s in splittings]
    x = rng.uniform(starts, stops, size=(10000, bdt.numberOfFeatures))
    x = x.astype(np.float32)
    sess = ort.InferenceSession(model.SerializeToString())
    out_onnx = sess.run(None, {"input": x})[0].ravel().tolist()
    clf = Classifier()
    clf.load("data/FastBDTv5.txt")
    out_fastbdt = clf.predict(x).ravel().tolist()
    assert out_onnx == pytest.approx(out_fastbdt, abs=1e-6)
