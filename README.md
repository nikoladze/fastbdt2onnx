> **Note**
> This is in an early stage and currently only tested against one example model - please validate the resulting ONNX models 
> **Note**
> NaN input values are currently not treated correctly yet

# Example usage
Either via command line tool `fastbdt2onnx`, e.g. using [uvx](https://docs.astral.sh/uv/concepts/tools/#the-uv-tool-interface):

```bash
uvx git+https://github.com/nikoladze/fastbdt2onnx <fastbdt-txt-file> <onnx-model-file>
```

Or via the python API:

```bash
pip install git+https://github.com/nikoladze/fastbdt2onnx
```

```python
import onnx
from fastbdt2onnx import convert
model_proto = convert(fastbdt_textfile)
onnx.save(model_proto, onnx_outputfile)
```

# Development Setup (only needed for tests)

``` bash
git submodule init
git submodule update
cd FastBDT
cmake .
make

cd ..
uv sync
```

# Run tests

``` bash
uv run --group test pytest tests.py
```
