"""
Some dataclasses for a structured representation of FastBDT parameters and
reading it from a sequence of space separated number strings

Mainly following the code from

- https://github.com/thomaskeck/FastBDT/blob/master/include/FastBDT_IO.h
- https://github.com/thomaskeck/FastBDT/blob/master/src/FastBDT_IO.cxx
"""

from itertools import chain
import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def next_non_space(tokens):
    while (token := next(tokens)).isspace():
        continue
    return token


def read(tokens, conv=int):
    logger.debug(f"read {conv}")
    if conv is bool:
        conv = lambda s: bool(int(s))
    return conv(next_non_space(tokens))


def read_vector(tokens, conv=float):
    logger.debug(f"read vector<{conv}>")
    size = int(next_non_space(tokens))
    return [conv(next_non_space(tokens)) for i in range(size)]


def read_vector_feature_binning(tokens):
    logger.debug(f"read vector of feature binning")
    out = []
    size = read(tokens, int)
    for i in range(size):
        n_levels = read(tokens, int)
        binning = read_vector(tokens, float)
        out.append((n_levels, binning))
    return out


def write_vector(vector):
    yield str(len(vector))
    for x in vector:
        if isinstance(x, bool):
            x = int(x)
        yield str(x)
    yield "\n"


def write_vector_feature_binning(vec_feature_binning):
    yield str(len(vec_feature_binning))
    for n_levels, binning in vec_feature_binning:
        yield str(n_levels)
        yield from write_vector(binning)
    yield "\n"


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

    def to_tokens(self):
        yield str(self.feature)
        yield str(self.index)
        yield str(self.valid)
        yield str(self.gain)
        yield "\n"


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

    def to_tokens(self):
        yield str(len(self.cuts))
        yield "\n"
        for cut in self.cuts:
            yield from cut.to_tokens()
            yield "\n"
        yield from write_vector(self.boost_weights)
        yield "\n"
        yield from write_vector(self.purities)
        yield "\n"
        yield from write_vector(self.nEntries)
        yield "\n"


@dataclass
class Forest:
    f0: float
    shrinkage: float
    transform2probability: bool
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

    def to_tokens(self):
        yield str(self.f0)
        yield "\n"
        yield str(self.shrinkage)
        yield "\n"
        yield str(int(self.transform2probability))
        yield "\n"
        yield str(len(self.trees))
        yield "\n"
        for tree in self.trees:
            yield from tree.to_tokens()
            yield "\n"


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
    was_read_from_forest: bool

    @classmethod
    def from_tokens(cls, tokens):
        first_token = next(tokens)
        try:
            version = int(first_token)
        except ValueError:
            logger.info("Couldn't read version - trying to read as forest-only file")
            forest = Forest.from_tokens(chain([first_token], tokens))
            logger.info("Successfully read as forest-only file!")
            return cls.from_forest(forest)
        return cls(
            version=version,
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
            was_read_from_forest=False,
        )

    @classmethod
    def from_file(cls, f):
        return cls.from_tokens(iter_tokens(f))

    @classmethod
    def from_string(cls, s):
        return cls.from_tokens(iter(s.split()))

    @classmethod
    def from_forest(cls, forest):
        "to read from older files that only contain the forest"
        max_index = max([cut.feature for tree in forest.trees for cut in tree.cuts])
        n_features = max_index + 1
        return cls(
            version=1,
            n_trees=len(forest.trees),
            depth=int(math.log2(len(forest.trees[0].cuts) + 1)),
            binning=[],
            shrinkage=forest.shrinkage,
            subsample=1,
            sPlot=False,
            flatnessLoss=-1,
            purityTransformation=[],
            transform2probability=forest.transform2probability,
            featureBinning=[],
            purityBinning=[],
            numberOfFeatures=n_features,
            numberOfFinalFeatures=n_features,
            numberOfFlatnessFeatures=0,
            can_use_fast_forest=True,
            forest=forest,
            binned_forest=Forest(0, 1, True, []),
            was_read_from_forest=True,
        )

    def to_tokens(self):
        # fmt: off
        yield str(self.version); yield "\n"
        yield str(self.n_trees); yield "\n"
        yield str(self.depth); yield "\n"
        yield from write_vector(self.binning); yield "\n"
        yield str(self.shrinkage); yield "\n"
        yield str(self.subsample); yield "\n"
        yield str(int(self.sPlot)); yield "\n"
        yield str(self.flatnessLoss); yield "\n"
        yield from write_vector(self.purityTransformation); yield "\n"
        yield str(int(self.transform2probability)); yield "\n"
        yield from write_vector_feature_binning(self.featureBinning); yield "\n"
        yield from write_vector(self.purityBinning); yield "\n"
        yield str(self.numberOfFeatures); yield "\n"
        yield str(self.numberOfFinalFeatures); yield "\n"
        yield str(self.numberOfFlatnessFeatures); yield "\n"
        yield str(int(self.can_use_fast_forest)); yield "\n"
        yield from self.forest.to_tokens(); yield "\n"
        yield from self.binned_forest.to_tokens(); yield "\n"
        # fmt: on

    def to_file(self, file):
        space = False
        for token in self.to_tokens():
            if token == "\n":
                space = False
            if space:
                file.write(" ")
            file.write(token)
            if token != "\n":
                space = True
