from __future__ import annotations
from functools import lru_cache
from typing import Any, Optional, Sequence

import attr
import funcy as fn
import dfa
import numpy as np
from dfa import DFA
from dfa.utils import find_subset_counterexample, find_equiv_counterexample
from dfa.utils import minimize
from dfa_identify import find_dfa, find_dfas
from diss.dfa_identify.decompose import enumerate_pareto_frontier_then_beyond

from diss import LabeledExamples, ConceptIdException
from diss import DemoPrefixTree as PrefixTree
from diss.learn import surprisal
from diss.concept_classes.dfa_concept import DFAConcept
from diss.concept_classes.dfa_decomposition_concept import DFAProductConcept

from functools import reduce
from more_itertools import interleave_longest, interleave
from collections import deque


__all__ = ['to_concept', 'ignore_white']


def transition(s, c):
    if c == 'red':
        return s | 0b01
    elif c == 'yellow':
        return s | 0b10
    return s


ALPHABET = frozenset({'red', 'yellow', 'blue', 'green'})


PARTIAL_DFA =  DFA(
    start=0b00,
    inputs=ALPHABET,
    label=lambda s: s == 0b10,
    transition=transition
)


def ignore_white(path):
    return tuple(x for x in path if x != 'white')


def dont_count(aps):
    for curr, prev in fn.with_prev(aps):
        if curr == prev:
            continue
        yield curr

def subset_check_wrapper(dfa_candidate):
    partial = partial_dfa(dfa_candidate.inputs)
    return find_subset_counterexample(dfa_candidate, partial) is None



BASE_EXAMPLES = LabeledExamples(
    positive=[
        ('yellow',),
        ('yellow', 'yellow'),
    ],
    negative=[
        (),
        ('blue',),
        ('blue', 'blue'),
        ('blue', 'green'),
        ('blue', 'red'),
        ('blue', 'red', 'green'),
        ('blue', 'red', 'green', 'yellow'),
        ('blue', 'red', 'yellow'),
        ('red',),
        ('red', 'blue'),
        ('red', 'blue', 'yellow'),
        ('red', 'green'),
        ('red', 'green', 'green'),
        ('red', 'green', 'green', 'yellow'),
        ('red', 'green', 'yellow'),
        ('red', 'red'),
        ('red', 'red', 'green'),
        ('red', 'red', 'green', 'yellow'),
        ('red', 'red', 'yellow'),
        ('red', 'yellow'),
        ('red', 'yellow', 'green'),
        ('red', 'yellow', 'green', 'yellow'),
        ('yellow', 'red'),
        ('yellow', 'red', 'green'),
        ('yellow', 'red', 'green', 'yellow'),
        ('yellow', 'red', 'yellow'),
        ('yellow', 'yellow', 'red')
    ]
)


@lru_cache
def find_dfas2(accepting, rejecting, alphabet, order_by_stutter=False, N=20):
    reach1 = set.union(*map(set, accepting)) if accepting else set()
    avoid = set.union(*map(set, rejecting)) if rejecting else set()
    reach2 = reach1 - avoid

    for x in set(avoid):
        problem_words = (w for w in accepting if x in w)
        for word in problem_words:
            prefix = word[:word.index(x)]
            if len(reach2 & set(prefix)) == 0:
                avoid.remove(x)
                break
    avoid -= reach1  # Make sure now to kill anything in accepting.

    print(f'{avoid=}')
    if avoid:
        avoid_lang = DFA(
            start=True, inputs=alphabet, label=bool,
            transition=lambda s, c: s and (c not in avoid)
        )
        assert all(not (set(w) & avoid) for w in accepting)
        rejecting = {w for w in rejecting if not (set(w) & avoid)}

    dfas = find_dfas(
        accepting,
        rejecting,
        alphabet=alphabet,
        order_by_stutter=order_by_stutter,
    )
    if avoid:
        dfas = (minimize(lang & avoid_lang) for lang in dfas)

    return fn.take(N, dfas)


@lru_cache
def augment(self: PartialDFAIdentifier, data: LabeledExamples) -> LabeledExamples:
    data = data.map(ignore_white) @ self.base_examples

    for i in range(20):
        tests = find_dfas2(
            data.positive,
            data.negative,
            order_by_stutter=True,
            alphabet=self.partial.dfa.inputs,
        )
        new_data = LabeledExamples()
        for test in tests:
            assert test is not None
            ce = self.subset_ce(test)
            if ce is None:
                continue
            new_data @= LabeledExamples(negative=[ce])
            partial = self.partial_dfa(test.inputs)
            for k, lbl in enumerate(partial.transduce(ce)):
                prefix = ce[:k]
                if not lbl:
                    new_data @= LabeledExamples(negative=[prefix])
            data @= new_data

        if new_data.size == -1:
            break
    return data

def remove_partial_examples(partials: list[DFA], data: LabeledExamples):
    partial_conj = dfa.utils.minimize(reduce(lambda x, y: x & y, partials))

    # remove data that is already labeled as reject (i.e. False)
    augmented_neg = filter(partial_conj.label, data.negative)

    return LabeledExamples(negative=augmented_neg, positive=data.positive)

@attr.frozen
class PartialProductDFAIdentifier:
    partials: list[DFA] = None
    base_examples: LabeledExamples = LabeledExamples()
    alphabet: set = None

    def __call__(self, data: LabeledExamples, concept: DFAProductConcept) -> DFAProductConcept:

        # # filter out white
        # new_pos = []
        # for t in data.positive:
        #     new_t = []
        #     for c in t:
        #         if c != "white":
        #             new_t.append(c)
        #     new_pos.append(tuple(new_t))

        # new_neg = []
        # for t in data.negative:
        #     new_t = []
        #     for c in t:
        #         if c != "white":
        #             new_t.append(c)
        #     new_neg.append(tuple(new_t))

        # data = LabeledExamples(positive=new_pos, negative=new_neg)

        # remove the known partial dfas from the product
        if self.partials is not None:
            num_partials = len(self.partials)
            if concept is not None:
                augmented_dfas = [dfa_concept.dfa for dfa_concept in concept.dfa_concepts[num_partials:]]
                reference = DFAProductConcept.from_dfas(augmented_dfas)
            else:
                reference = concept
            data = remove_partial_examples(self.partials, data)
        else:
            reference = concept

        data = data @ self.base_examples

        concept = DFAProductConcept.from_examples(
            data=data,
            order_by_stutter=True,
            temp=1,
            alphabet=self.alphabet,
            ref=reference
        ) 

        if self.partials is not None:
            augmented_dfas = self.partials + [dfa_concept.dfa for dfa_concept in concept.dfa_concepts] 
            return DFAProductConcept.from_dfas(augmented_dfas)
        else:
            return concept


@attr.frozen
class PartialDFAIdentifier:
    partial: DFAConcept = attr.ib(converter=DFAConcept.from_dfa)
    base_examples: LabeledExamples = LabeledExamples()
    try_reach_avoid: bool = False

    def partial_dfa(self, inputs) -> DFA:
        assert inputs <= self.partial.dfa.inputs
        return attr.evolve(self.partial.dfa, inputs=inputs)

    def subset_ce(self, candidate: DFA) -> Optional[Sequence[Any]]:
        partial = self.partial_dfa(candidate.inputs)
        return find_subset_counterexample(candidate, partial)

    def is_subset(self, candidate: DFA) -> Optional[Sequence[Any]]:
        return self.subset_ce(candidate) is None

    def __call__(self, data: LabeledExamples, concept: DFAConcept) -> DFAConcept:
        reference = concept

        data = augment(self, data)

        concept = DFAConcept.from_examples(
            data=data,
            filter_pred=self.is_subset,
            alphabet=self.partial.dfa.inputs,
            find_dfas=find_dfas2,
            order_by_stutter=True,
            temp=1,
            ref=reference
        ) 

        # Adjust size to account for subset information.
        return attr.evolve(concept, size=concept.size - self.partial.size)


def enumerative_search(
    demos: Demos, 
    identifer: PartialDFAIdentifier(),
    to_chain: MarkovChainFact,
    competency: CompetencyEstimator,
    n_iters: int = 25,
    size_weight: float = 1,
    surprise_weight: float = 1,
):
    tree = PrefixTree.from_demos(demos)
    weights = np.array([size_weight, surprise_weight])
    data = augment(identifer, LabeledExamples())
    dfas = find_dfas(
        accepting=data.positive,
        rejecting=data.negative,
        order_by_stutter=True,
        allow_unminimized=True,
        alphabet=identifer.partial.dfa.inputs
    )
    dfas = (attr.evolve(d, outputs={True, False}) for d in dfas)
    dfas = filter(identifer.is_subset, dfas)
    dfas = map(minimize, dfas)
    dfas = fn.distinct(dfas)

    # Convert to representation class.
    ref_size = identifer.partial.size
    concepts = map(DFAConcept.from_dfa, dfas)
    concepts = (attr.evolve(c, size=c.size - ref_size) for c in concepts)
    print(f'Enumerating {n_iters} DFAs in lexicographic order...')
    concepts = fn.take(n_iters, concepts)
    print(f'Sorting by size')
    concepts = sorted(concepts, key=lambda c: c.size)
    for concept in concepts:
        chain = to_chain(concept, tree, competency(concept, tree))
        metadata = {
            'energy': weights @ [concept.size, surprisal(chain, tree)],
        }
 
        yield LabeledExamples(), concept, metadata

def bfs(data, num_dfas, alphabet=None):
    min_dfa_size = [2]*num_dfas
    size_q = deque()
    size_q.append(min_dfa_size)
    while size_q:
        dfa_sizes = size_q.popleft()
        print(dfa_sizes)
        dfa_gen = find_dfa_decompositions(
            accepting=data.positive,
            rejecting=data.negative,
            num_dfas=num_dfas,
            dfa_sizes=dfa_sizes,
            order_by_stutter=True,
            allow_unminimized=False,
            alphabet=alphabet
        )
        try:
            # yield the dfa from this generator
            yield dfa_gen #, sum(dfa_sizes)
        except StopIteration:
            pass
        for i in range(num_dfas):
            new_dfa_sizes = list(dfa_sizes)
            new_dfa_sizes[i] += 1
            nondecreasing = all(new_dfa_sizes[i] <= new_dfa_sizes[i+1] for i in range(len(new_dfa_sizes) - 1))
            not_in_queue = new_dfa_sizes not in size_q

            # we want to avoid making symmetric solves, so only append the sizes that are ordered in increasing size
            if nondecreasing and not_in_queue:
                size_q.append(new_dfa_sizes)

# def exhaust_pareto_frontier(data, num_dfa_upper, alphabet=None):
#     dfa_state_sum = 2
#     while True:
#         for num_dfas in range(num_dfa_upper+1):
#             # get all valid generators for dfa decompositions with num_dfas dfas whose states sum to dfa_state_sum
#             for m in itertools


#         dfa_state_sum += 1

def get_next_smallest(dfas_gens):
    prev_sizes = [0] * len(dfas_gens)
    while True:
        i = np.argmin(prev_sizes)
        next_dfas = next(dfas_gens[i])
        yield from next_dfas
        prev_sizes[i] = sum([len(dfa.states()) for dfa in next_dfas])

def decomposition_enumerative_search(
    demos: Demos, 
    identifer: PartialProductDFAIdentifier(),
    to_chain: MarkovChainFact,
    competency: CompetencyEstimator,
    n_iters: int = 25,
    size_weight: float = 1,
    surprise_weight: float = 1,
    num_dfas_upper: int = 3,
):
    tree = PrefixTree.from_demos(demos)
    weights = np.array([size_weight, surprise_weight])
    data = identifer.base_examples
    if identifer.partials is not None:
        data = remove_partial_examples(identifer.partials, data)

    # dfa_gens = [bfs(data, num_dfas, identifer.alphabet) for num_dfas in range(1,num_dfas_upper+1)]
    # dfas = get_next_smallest(dfa_gens)
    dfa_gens = [enumerate_pareto_frontier_then_beyond(data.positive, 
                data.negative, 
                alphabet=identifer.alphabet, 
                order_by_stutter=True,
                num_dfas=num_dfas) for num_dfas in range(1,num_dfas_upper+1)]
    dfas = interleave(*dfa_gens)
    # dfas = (attr.evolve(d, outputs={True, False}) for d in dfas)
    # dfas = filter(identifer.is_subset, dfas)
    dfas = map(lambda x : [minimize(dfa) for dfa in x], dfas)
    # pass to tuple so that we can hash
    dfas = map(tuple, dfas)
    dfas = fn.distinct(dfas)

    # add the partial dfas 
    if identifer.partials is not None:
        dfas = map(lambda x : identifer.partials + list(x), dfas)

    # Convert to representation class.
    # ref_size = identifer.partial.size
    concepts = map(DFAProductConcept.from_dfas, dfas)
    # concepts = (attr.evolve(c, size=c.size - ref_size) for c in concepts)
    print(f'Enumerating {n_iters} DFAs in lexicographic order...')
    concepts = fn.take(n_iters, concepts)
    print(f'Sorting by size')
    concepts = sorted(concepts, key=lambda c: c.size)
    for concept in concepts:
        chain = to_chain(concept, tree, competency(concept, tree))
        metadata = {
            'energy': weights @ [concept.size, surprisal(chain, tree)],
        }
 
        yield LabeledExamples(), concept, metadata
