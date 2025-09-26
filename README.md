# Development Setup

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
