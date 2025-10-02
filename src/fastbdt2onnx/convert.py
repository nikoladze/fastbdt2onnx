import argparse
import io
from contextlib import contextmanager
from enum import IntEnum
from pathlib import Path
from typing import IO

import onnx
from onnx import TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor,
    make_tensor_value_info,
)
from onnx.onnx_ml_pb2 import ModelProto

from fastbdt2onnx.bdt import BDT, Forest, iter_tokens


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


def _tensor_1d(tensor_type, name, values):
    return make_tensor(name, tensor_type, (len(values),), values)


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


def _get_onnx_model(
    number_of_inputs,
    leaf_weights,
    nodes_falseleafs,
    nodes_trueleafs,
    nodes_falsenodeids,
    nodes_truenodeids,
    nodes_featureids,
    nodes_splits,
    tree_roots,
    # post transform seems to be ignored for 1-class outputs
    # (https://github.com/microsoft/onnxruntime/issues/24862)
    # so we apply sigmoid manually later
    post_transform=PostTransform.NONE,
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
        nodes_missing_value_tracks_true=[1 for __ in nodes_splits],
        nodes_hitrates=None,
        aggregate_function=AggregateFunction.SUM,
        post_transform=post_transform,
        tree_roots=tree_roots,
        nodes_modes=_tensor_1d(TensorProto.UINT8, "nodes_modes", nodes_modes),
        nodes_featureids=nodes_featureids,
        nodes_splits=_tensor_1d(TensorProto.FLOAT, "nodes_splits", nodes_splits),
        nodes_truenodeids=nodes_truenodeids,
        nodes_trueleafs=nodes_trueleafs,
        nodes_falsenodeids=nodes_falsenodeids,
        nodes_falseleafs=nodes_falseleafs,
        leaf_targetids=leaf_targetids,
        leaf_weights=_tensor_1d(TensorProto.FLOAT, "leaf_weights", leaf_weights),
    )
    sigmoid = make_node(
        "Sigmoid",
        inputs=["forest"],
        outputs=["output"],
        name="Sigmoid",
    )
    graph = make_graph(
        [forest, sigmoid],
        "FastBDT",
        [
            make_tensor_value_info(
                "input",
                TensorProto.FLOAT,
                [None, number_of_inputs],
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


def _get_tree_ensemble_attrs(bdt):
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

    # work around https://github.com/microsoft/onnxruntime/issues/24679
    # otherwise the leaf and node index of the second tree root will coincide
    # and trigger the issue.
    # The issue is fixed by now (https://github.com/microsoft/onnxruntime/pull/25410)
    # so once this lands in a release the workaround may be removed eventually
    # (start with leaf_offset = 0)
    leaf_offset = 2
    leaf_weights.append(float("nan"))
    leaf_weights.append(float("nan"))

    node_offset = 0
    f0 = bdt.forest.f0
    for tree in bdt.forest.trees:
        tree_roots.append(node_offset)

        # we loop over non-duplicated node indices
        # so at various places factors of 2 appear
        for node, cut in enumerate(tree.cuts):
            is_terminal = node >= n_internal_nodes

            # add the NaN node:
            # non-NaN values will go to the false branch, which is then the
            # actual node. NaN values will go to the true branch since we will
            # set `nodes_missing_value_tracks_true` for all nodes
            nodes_trueleafs.append(1)
            nodes_falseleafs.append(0)
            # NaNs go to leaf with same index as node (in non-duplicated index space)
            nodes_truenodeids.append(node + leaf_offset)
            # The rest goes to the next actual node (we need to multiply by 2)
            nodes_falsenodeids.append(2 * node + 1 + node_offset)
            # Splitting on NaN ensures all non-NaN values go to false
            nodes_splits.append(float("nan"))
            nodes_featureids.append(cut.feature)

            if (not cut.valid) or is_terminal:
                # go to leaf after this
                nodes_falseleafs.append(1)
                nodes_trueleafs.append(1)
                if cut.valid:
                    # terminal node:
                    # same index as for non-terminal, but index into leafs,
                    # so with leaf offset and index not duplicated
                    nodes_falsenodeids.append(2 * (node + 1) - 1 + leaf_offset)
                    nodes_truenodeids.append(2 * (node + 1) + leaf_offset)
                else:
                    # "invalid" node, meaning tree stops here and goes to leaf
                    # TODO: could try to optimize this by directly going to the
                    # leaf from the node that lead use here and prune all nodes
                    # following from here
                    nodes_falsenodeids.append(node + leaf_offset)
                    nodes_truenodeids.append(node + leaf_offset)
            else:
                # internal node, go to next node depending on the splitting
                # criterion we need an additional factor of 2 in the node
                # indices since we duplicated the nodes for the NaN nodes
                nodes_falseleafs.append(0)
                nodes_trueleafs.append(0)
                nodes_falsenodeids.append(2 * (2 * (node + 1) - 1) + node_offset)
                nodes_truenodeids.append(2 * (2 * (node + 1)) + node_offset)
            nodes_featureids.append(cut.feature)
            nodes_splits.append(cut.index)

        # since we have all nodes and leafs in one big list
        # we need to count offsets up
        node_offset += 2 * len(tree.cuts)
        leaf_offset += len(tree.boost_weights)
        # factor in shrinkage and factor of 2 used in FastBDT pre-sigmoid
        weights = [w * bdt.shrinkage * 2 for w in tree.boost_weights]
        if f0:
            # add f0 (also with factor of 2) to first tree
            weights = [w + 2 * f0 for w in weights]
            f0 = None
        leaf_weights.extend(weights)

    return dict(
        leaf_weights=leaf_weights,
        nodes_falseleafs=nodes_falseleafs,
        nodes_trueleafs=nodes_trueleafs,
        nodes_falsenodeids=nodes_falsenodeids,
        nodes_truenodeids=nodes_truenodeids,
        nodes_featureids=nodes_featureids,
        nodes_splits=nodes_splits,
        tree_roots=tree_roots,
    )


def convert(file: str | Path | IO | bytes, from_forest=False) -> ModelProto:
    if not from_forest:
        with _read_file(file) as f:
            bdt = BDT.from_file(f)
    else:
        with _read_file(file) as f:
            forest = Forest.from_tokens(iter_tokens(f))
            bdt = BDT.from_forest(forest)
    return _get_onnx_model(
        number_of_inputs=bdt.numberOfFeatures,
        **_get_tree_ensemble_attrs(bdt),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fastbdt_textfile")
    parser.add_argument("onnx_outputfile")
    parser.add_argument(
        "--from-forest",
        action="store_true",
        help="Read old FastBDT file with only forest information",
    )
    args = parser.parse_args()
    model = convert(args.fastbdt_textfile)
    onnx.save(model, args.onnx_outputfile)


if __name__ == "__main__":
    main()
