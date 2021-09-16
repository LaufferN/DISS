from __future__ import annotations

import random
from typing import Any, Callable, Iterable, Optional, Protocol

import attr

from diss import AnnotatedMarkovChain as MarkovChain
from diss import Demos, Path
from diss.prefix_tree import DemoPrefixTree as PrefixTree


Examples = frozenset[Any]


@attr.frozen
class LabeledExamples:
    positive: Examples = attr.ib(converter=frozenset, factory=frozenset)
    negative: Examples = attr.ib(converter=frozenset, factory=frozenset)

    @property
    def size(self) -> int:
        return self.dist(LabeledExamples())

    def __matmul__(self, other: LabeledExamples) -> LabeledExamples:
        return LabeledExamples(
            positive=(self.positive - other.negative) | other.positive,
            negative=(self.negative - other.positive) | other.negative,
        )

    def dist(self, other: LabeledExamples) -> int:
        pos_delta = self.positive ^ other.positive
        neg_delta = self.negative ^ other.negative
        return len(pos_delta) + len(neg_delta) - len(pos_delta & neg_delta)


class Concept(Protocol):
    size: int

    def __contains__(self, val: Any) -> bool:
        ...


###############################################################################
#                              Guided Search 
###############################################################################

Concept2MC = Callable[[Concept, PrefixTree], MarkovChain]
Identify = Callable[[LabeledExamples], Concept]
ExampleSampler = Callable[[MarkovChain, PrefixTree], LabeledExamples]


def sample_examples(chain: MarkovChain, tree: PrefixTree) -> LabeledExamples:
    # TODO: Compute gradient of surprisal w.r.t. conform/deviate leaves.
    #       Note: surprisal if taken as function of variables indexed
    #             by each (non-exhausted) node of the tree.
    # TODO: Sample node according to gradient.
    # TODO: Find path to node.
    # TODO: If interior (non-exhaused) node, change path[-1] to deviate.
    # TODO: Look at sign of corresponding entry in gradient to give label.
    # TODO: Extend path to node.
    # TODO: If extension fails assign 0 weight to that node and repeat.
    # TODO: Return Labeled Examples with path.
    ...


def search(
    demos: Demos, 
    to_chain: Concept2MC, 
    to_concept: Identify,
    sample_examples: ExampleSampler = sample_examples,
) -> Iterable[Concept]:
    """Perform demonstration informed gradiented guided search."""
    tree = PrefixTree.from_demos(demos)
    example_state = LabeledExamples()

    while True:
        concept = to_concept(example_state)
        yield concept
        chain = to_chain(concept, tree)
        example_state @= sample_examples(chain, tree)
