import argparse
import io
from pathlib import Path
from typing import IO
from enum import IntEnum
from contextlib import contextmanager

import onnx
from onnx import TensorProto
from onnx.helper import (
    make_model,
    make_node,
    make_tensor_value_info,
    make_graph,
    make_tensor,
)
from onnx.onnx_ml_pb2 import ModelProto

from fastbdt2onnx.bdt import BDT


@contextmanager
def _read_file(file: str | Path | IO | bytes) -> IO:
    """
    Get file-like object from a path, file-like object, or bytes.

    Args:
        file: File input which can be:
            - str or pathlib.Path: path to a file
            - file-like object: must implement `.readline()`
            - bytes: raw file content
    Returns:
        file: The file-like object
    """
    if isinstance(file, (str, Path)):
        with open(file, "r") as f:
            yield f
    elif isinstance(file, bytes):
        yield io.BytesIO(file)
    elif hasattr(file, "readline"):
        yield file
    else:
        raise TypeError(
            f"Unsupported input type: {type(file).__name__}. "
            "Must be str, Path, file-like object, or bytes."
        )


def _to_tensor(tensor_type, **kwargs):
    for k, v in kwargs.items():
        return make_tensor(
            k,
            tensor_type,
            (len(v),),
            v,
        )


class NodeMode(IntEnum):
    BRANCH_LEQ = 0
    BRANCH_LT = 1
    BRANCH_GTE = 2
    BRANCH_GT = 3
    BRANCH_EQ = 4
    BRANCH_NEQ = 5
    BRANCH_MEMBER = 6


class PostTransform(IntEnum):
    NONE = 0
    SOFTMAX = 1
    LOGISTIC = 2
    SOFTMAX_ZERO = 3
    PROBIT = 4


class AggregateFunction(IntEnum):
    AVERAGE = 0
    SUM = 1
    MIN = 2
    MAX = 3


def _to_onnx(
    bdt,
    leaf_weights,
    nodes_falseleafs,
    nodes_trueleafs,
    nodes_falsenodeids,
    nodes_truenodeids,
    nodes_featureids,
    nodes_splits,
    tree_roots,
    # this is just metadata, we still have to add a sigmoid node
    post_transform=PostTransform.LOGISTIC,
):
    nodes_modes = [NodeMode.BRANCH_GTE for __ in nodes_splits]
    leaf_targetids = [0 for __ in leaf_weights]
    forest = make_node(
        "TreeEnsemble",
        ["input"],
        ["forest"],
        domain="ai.onnx.ml",
        n_targets=1,
        membership_values=None,
        nodes_missing_value_tracks_true=None,
        nodes_hitrates=None,
        aggregate_function=AggregateFunction.SUM,
        post_transform=post_transform,
        tree_roots=tree_roots,
        nodes_modes=_to_tensor(TensorProto.UINT8, nodes_modes=nodes_modes),
        nodes_featureids=nodes_featureids,
        nodes_splits=_to_tensor(TensorProto.FLOAT, nodes_splits=nodes_splits),
        nodes_truenodeids=nodes_truenodeids,
        nodes_trueleafs=nodes_trueleafs,
        nodes_falsenodeids=nodes_falsenodeids,
        nodes_falseleafs=nodes_falseleafs,
        leaf_targetids=leaf_targetids,
        leaf_weights=_to_tensor(TensorProto.FLOAT, leaf_weights=leaf_weights),
    )
    f0 = make_node(
        "Constant",
        inputs=[],
        outputs=["f0"],
        value=_to_tensor(TensorProto.FLOAT, F0=[bdt.forest.f0]),
    )
    add_f0 = make_node(
        "Add",
        inputs=["forest", "f0"],
        outputs=["add_f0"],
        name="AddF0",
    )
    two = make_node(
        "Constant",
        inputs=[],
        outputs=["two"],
        value=_to_tensor(TensorProto.FLOAT, TWO=[2]),
    )
    twice = make_node(
        "Mul",
        inputs=["add_f0", "two"],
        outputs=["twice"],
        name="Twice",
    )
    sigmoid = make_node(
        "Sigmoid",
        inputs=["twice"],
        outputs=["output"],
        name="Sigmoid",
    )
    graph = make_graph(
        [forest, f0, add_f0, two, twice, sigmoid],
        "FastBDT",
        [
            make_tensor_value_info(
                "input",
                TensorProto.FLOAT,
                [None, bdt.numberOfFeatures],
            )
        ],
        [
            make_tensor_value_info(
                "output",
                TensorProto.FLOAT,
                [None, 1],
            ),
        ],
    )
    return make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("ai.onnx.ml", 5),
            onnx.helper.make_opsetid("", 21),
        ],
        ir_version=10,
    )


def convert(file: str | Path | IO | bytes) -> ModelProto:
    with _read_file(file) as f:
        bdt = BDT.from_file(f)

    # todo: proper error messages for these
    assert bdt.can_use_fast_forest
    assert bdt.transform2probability

    leaf_weights = []
    nodes_falseleafs = []
    nodes_trueleafs = []
    nodes_falsenodeids = []
    nodes_truenodeids = []
    nodes_featureids = []
    nodes_splits = []
    tree_roots = []

    n_leafs = 2 ** (bdt.depth)
    n_nodes = len(bdt.forest.trees[0].cuts)
    n_terminal_nodes = n_leafs // 2
    n_internal_nodes = n_nodes - n_terminal_nodes

    node_offset = 0
    leaf_offset = 0
    for tree in bdt.forest.trees:
        tree_roots.append(node_offset)
        for node, cut in enumerate(tree.cuts):
            is_terminal = node >= n_internal_nodes
            if (not cut.valid) or is_terminal:
                nodes_falseleafs.append(1)
                nodes_trueleafs.append(1)
                if cut.valid:
                    # same index as for non-terminal, but index into leafs, so with leaf offset
                    nodes_falsenodeids.append(2 * (node + 1) + leaf_offset - 1)
                    nodes_truenodeids.append(2 * (node + 1) + 1 + leaf_offset - 1)
                else:
                    nodes_falsenodeids.append(node + leaf_offset)
                    nodes_truenodeids.append(node + leaf_offset)
            else:
                nodes_falseleafs.append(0)
                nodes_trueleafs.append(0)
                nodes_falsenodeids.append(2 * (node + 1) + node_offset - 1)
                nodes_truenodeids.append(2 * (node + 1) + 1 + node_offset - 1)
            nodes_featureids.append(cut.feature)
            nodes_splits.append(cut.index)
        node_offset += len(tree.cuts)
        leaf_offset += len(tree.boost_weights)
        leaf_weights += [w * bdt.shrinkage for w in tree.boost_weights]

    return _to_onnx(
        bdt,
        leaf_weights=leaf_weights,
        nodes_falseleafs=nodes_falseleafs,
        nodes_trueleafs=nodes_trueleafs,
        nodes_falsenodeids=nodes_falsenodeids,
        nodes_truenodeids=nodes_truenodeids,
        nodes_featureids=nodes_featureids,
        nodes_splits=nodes_splits,
        tree_roots=tree_roots,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fastbdt_textfile")
    parser.add_argument("onnx_outputfile")
    args = parser.parse_args()
    model = convert(args.fastbdt_textfile)
    onnx.save(model, args.onnx_outputfile)


if __name__ == "__main__":
    main()
