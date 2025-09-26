"""
Some dataclasses for a structured representation of FastBDT parameters and
reading it from a sequence of space separated number strings

Mainly following the code from

- https://github.com/thomaskeck/FastBDT/blob/master/include/FastBDT_IO.h
- https://github.com/thomaskeck/FastBDT/blob/master/src/FastBDT_IO.cxx
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger()


def read(tokens, conv=int):
    logger.debug(f"read {conv}")
    return conv(next(tokens))


def read_vector(tokens, conv=float):
    logger.debug(f"read vector<{conv}>")
    size = int(next(tokens))
    return [conv(next(tokens)) for i in range(size)]


def read_vector_feature_binning(tokens):
    logger.debug(f"read vector of feature binning")
    out = []
    size = read(tokens, int)
    for i in range(size):
        n_levels = read(tokens, int)
        binning = read_vector(tokens, float)
        out.append((n_levels, binning))
    return out


@dataclass
class Cut:
    feature: int
    index: ...
    gain: float
    valid: int

    @classmethod
    def from_tokens(cls, tokens, conv=float):
        logger.debug(f"Read Cut<{conv}>")
        feature = read(tokens, int)
        index = read(tokens, conv)
        valid = read(tokens, int)
        gain = read(tokens, float)
        return cls(feature, index, gain, valid)


@dataclass
class Tree:
    cuts: list[Cut]
    nEntries: int
    purities: float
    boost_weights: list[float]

    @classmethod
    def from_tokens(cls, tokens, conv=float):
        logger.debug(f"Read Tree<{conv}>")
        size = read(tokens, int)
        cuts = []
        for i in range(size):
            cuts.append(Cut.from_tokens(tokens, conv))
        boost_weights = read_vector(tokens, float)
        purities = read_vector(tokens, float)
        nEntries = read_vector(tokens, float)
        return cls(cuts, nEntries, purities, boost_weights)


@dataclass
class Forest:
    f0: float
    shrinkage: float
    transform2probability: list[bool]
    trees: list[Tree]

    @classmethod
    def from_tokens(cls, tokens, conv=float):
        logger.debug(f"Read Forest<{conv}>")
        f0 = read(tokens, float)
        shrinkage = read(tokens, float)
        transform2probability = read(tokens, bool)
        size = read(tokens, int)
        trees = []
        for i in range(size):
            trees.append(Tree.from_tokens(tokens, conv))
        return cls(f0, shrinkage, transform2probability, trees)


def iter_tokens(f):
    for line in f:
        for token in line.strip().split():
            yield token


@dataclass
class BDT:
    version: int
    n_trees: int
    depth: int
    binning: list[int]
    shrinkage: float
    subsample: float
    sPlot: bool
    flatnessLoss: float
    purityTransformation: list[bool]
    transform2probability: bool
    featureBinning: list[tuple[int, list[float]]]
    purityBinning: list[int]
    numberOfFeatures: int
    numberOfFinalFeatures: int
    numberOfFlatnessFeatures: int
    can_use_fast_forest: bool
    forest: Forest
    binned_forest: Forest

    @classmethod
    def from_tokens(cls, tokens):
        return cls(
            version=read(tokens, int),
            n_trees=read(tokens, int),
            depth=read(tokens, int),
            binning=read_vector(tokens, int),
            shrinkage=read(tokens, float),
            subsample=read(tokens, float),
            sPlot=read(tokens, bool),
            flatnessLoss=read(tokens, float),
            purityTransformation=read_vector(tokens, bool),
            transform2probability=read(tokens, bool),
            featureBinning=read_vector_feature_binning(tokens),
            purityBinning=read_vector(tokens, int),
            numberOfFeatures=read(tokens, int),
            numberOfFinalFeatures=read(tokens, int),
            numberOfFlatnessFeatures=read(tokens, int),
            can_use_fast_forest=read(tokens, bool),
            forest=Forest.from_tokens(tokens, float),
            binned_forest=Forest.from_tokens(tokens, int),
        )

    @classmethod
    def from_file(cls, f):
        return cls.from_tokens(iter_tokens(f))

    @classmethod
    def from_string(cls, s):
        return cls.from_tokens(s.split())
