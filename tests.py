import base64
from pathlib import Path
from urllib.parse import quote

import numpy as np
import onnxruntime as ort
import pytest
import requests
import uproot
import xmltodict
from PyFastBDT.FastBDT import Classifier

from fastbdt2onnx import convert
from fastbdt2onnx.bdt import BDT

URLS = {}
with open("data/urls.txt") as f:
    for line in f:
        identifier, base_url, url_path = line.strip().split()
        URLS[identifier] = (base_url, url_path)


@pytest.fixture(scope="session", params=URLS.keys())
def fastbdt_str(request):
    identifier = request.param
    root_path = Path(f"data/{identifier}.root")
    if not root_path.exists():
        base_url, url_path = URLS[identifier]
        full_url = f"{base_url}{quote(url_path)}"
        print(f"downloading {full_url}")
        res = requests.get(full_url)
        with open(root_path, "wb") as f:
            f.write(res.content)
    else:
        print(f"{str(root_path)} already exists!")
    with uproot.open(root_path) as f:
        obj = next(f.itervalues())
        xml_str = f"<root>{obj.member('m_data').split('?>')[1]}</root>"
        b64_str = xmltodict.parse(xml_str)["root"]["FastBDT_Weightfile"]
        return base64.b64decode(f"{b64_str}===").decode()


def assert_fastbdt_onnx_consistent(fastbdt_filename, use_nan=False, use_inf=False):
    rng = np.random.default_rng(42)
    with open(fastbdt_filename) as f:
        bdt = BDT.from_file(f)
    if bdt.was_read_from_forest:
        print("The BDT was read from forest-only information - writing a new file")
        fastbdt_filename = str(
            Path(fastbdt_filename).parent / "fastbdt_from_forest.txt"
        )
        with open(fastbdt_filename, "w") as f:
            bdt.to_file(f)
    model = convert(fastbdt_filename)
    splittings = [
        [
            cut.index
            for tree in bdt.forest.trees
            for cut in tree.cuts
            if cut.feature == i
        ]
        for i in range(bdt.numberOfFeatures)
    ]
    # random uniform numbers around the range of splittings if a variable is
    # used in any splitting (and non NaN), otherwise just 0-1
    starts = [np.nanmin(s).item() - 0.1 if s else 0 for s in splittings]
    stops = [np.nanmax(s).item() + 0.1 if s else 1 for s in splittings]
    starts = [s if not np.isnan(s) else 0 for s in starts]
    stops = [s if not np.isnan(s) else 0 for s in stops]
    x = rng.uniform(starts, stops, size=(10000, bdt.numberOfFeatures))
    x = x.astype(np.float32)
    if use_nan:
        x[rng.random(x.shape) < 0.05] = np.nan
    if use_inf:
        x[rng.random(x.shape) < 0.02] = np.inf
        x[rng.random(x.shape) < 0.02] = -np.inf
    sess = ort.InferenceSession(model.SerializeToString())
    out_onnx = sess.run(None, {"input": x})[0].ravel().tolist()
    clf = Classifier()
    clf.load(fastbdt_filename)
    out_fastbdt = clf.predict(x).ravel().tolist()
    assert out_onnx == pytest.approx(out_fastbdt, abs=1e-4)


@pytest.mark.parametrize("mode", ["with_nan", "no_nan", "with_inf"])
def test_consistent_fastbdt_onnx_belle2_payloads(fastbdt_str, tmp_path, mode):
    with open(tmp_path / "fastbdt.txt", "w") as f:
        f.write(fastbdt_str)
    assert_fastbdt_onnx_consistent(
        str(tmp_path / "fastbdt.txt"),
        use_nan=mode == "with_nan",
        use_inf=mode == "with_inf",
    )


@pytest.mark.parametrize("mode", ["with_nan", "no_nan", "with_inf"])
def test_consistent_fastbdt_onnx_example(mode):
    assert_fastbdt_onnx_consistent(
        "data/FastBDTv5.txt",
        use_nan=mode == "with_nan",
        use_inf=mode == "with_inf",
    )
