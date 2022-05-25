from __future__ import annotations

import random
from pprint import pformat
from typing import Callable 

import attr
import dfa
import funcy as fn
import numpy as np
from dfa.utils import find_equiv_counterexample, dfa2dict
from pysat.solvers import Minicard
from scipy.special import softmax

from diss import State, Path, LabeledExamples, ConceptIdException, MonitorState
from diss.concept_classes.dfa_concept import DFAConcept

from diss.dfa_identify.decompose import enumerate_pareto_frontier

from more_itertools import interleave_longest


__all__ = ['DFAConcept', 'Sensor']


DFA = dfa.DFA
Sensor = Callable[[dfa.State], dfa.Letter] 
ENUM_MAX = 100


def remove_stutter(graph: dfa.DFADict) -> None:
    for state, (_, kids) in graph.items():
        tokens = list(kids.keys())
        kids2 = {k: v for k, v in kids.items() if v != state}
        kids.clear()
        kids.update(kids2)

def measure_diffs(concepts: DFAProductConcepts, refs: DFAProductConcepts) -> float:
    # TODO change this?
    avg_measures = [np.average([measure_diff(c,r) for r in refs]) for c in concepts]
    return sum(avg_measures)

def measure_diff(concept: DFAConcept, ref: DFAConcept) -> float:
    lang = concept.dfa
    ref = ref.dfa

    if lang == ref:
        return 0  # Don't want to sample equivilent DFAs.

    graph, _ = dfa2dict(lang)
    graph_ref, _ = dfa2dict(ref)
    
    accepting = {(k, v) for k, (v, _) in graph.items()}
    accepting_ref = {(k, v) for k, (v, _) in graph_ref.items()}
    daccepting = len(accepting ^ accepting_ref) / 2
    dstates = abs(len(graph) - len(graph_ref))
    
    edges = set.union(*({(s, c, e) for c, e in trans.items()} for s, (_, trans) in graph.items()))
    edges_ref = set.union(*({(s, c, e) for c, e in trans.items()} for s, (_, trans) in graph_ref.items()))
    d_edges = len(edges ^ edges_ref) / 2
    
    size = dstates + np.log(len(graph)) * daccepting + d_edges * (2*np.log(len(graph)) + np.log(len(lang.inputs)))
    return size


def count_edges(graph: dfa.DFADict) -> int:
    count = 0
    for _, (_, kids) in graph.items():
        count += sum(1 for k in kids.values()) 
    return count

@attr.frozen
class DFAProductConcept:
    dfa_concepts: tuple[DFAConcept]
    size: float

    # def __hash__(self) -> int:
    #     return hash(self.dfa)

    # def __eq__(self, other) -> bool:
    #     return self.dfa == other.dfa

    # def __repr__(self) -> str:
    #     graph, start = dfa.dfa2dict(self.dfa)
    #     remove_stutter(graph)
    #     return f'{start}\n{pformat(graph)}'

    @staticmethod
    def from_examples(
            data: LabeledExamples, 
            # TODO wrap filter_pred to throw out products that have a size 1 sub dfa
            filter_pred: Callable[[list[DFA]], bool] = None,
            alphabet: frozenset = None,
            temp: float = 10,
            order_by_stutter=True,
            ref: DFAProductConcept = None,
            ) -> DFAProductConcept:

        if ref is not None:
            num_dfas = len(ref.dfa_concepts)
        else:
            num_dfas = 3


        lo = max(1, num_dfas-1)
        hi = num_dfas+1

        # perturb the num of dfas in the decomposition
        lang_perturbations = []
        for num_dfa_perturbed in range(lo, hi+1):
            lang_perturbations.append(enumerate_pareto_frontier(
                data.positive, data.negative, 
                alphabet=alphabet,
                order_by_stutter=order_by_stutter,
                num_dfas=num_dfa_perturbed,
                ))  # type: ignore
        
        # interleave the three types of perturbations
        langs = interleave_longest(*lang_perturbations)

        if filter_pred is not None:
            langs = filter(filter_pred, langs)
        langs = fn.take(ENUM_MAX, langs)
        if not langs:
            raise ConceptIdException

        # concepts = []
        # for i,lang in enumerate(langs):
        #     concepts.append(DFAProductConcept.from_dfas(lang))
        concepts = [DFAProductConcept.from_dfas(lang) for lang in langs]
        # TODO make a version of measure_diff and compare against ref
        sizes = np.array([c.size for c in concepts])
        weights = softmax(-sizes / temp)
        try:
            return random.choices(concepts, weights)[0]  # type: ignore
        except:
            return concepts[0]

    @staticmethod
    def from_dfas(lang: list[DFA]) -> DFAProductConcept:
        assert all(sub_lang.inputs is not None for sub_lang in lang)
        assert all(sub_lang.outputs <= {True, False} for sub_lang in lang)

        dfa_concepts = tuple([DFAConcept.from_dfa(sub_lang) for sub_lang in lang])

        N = len(dfa_concepts)

        dfa0 = dfa_concepts[0].dfa
        dfa_sizes = [len(dfa_c.dfa.states()) for dfa_c in dfa_concepts]

        b_Q1 = np.log(min(dfa_sizes))
        b_E = np.log(len(dfa0.inputs))

        # \sum_{i=1}(size(D_i)) - (n - 1)(2*b_E + 1) - (n - 1)(2*b_{Q_1}) 
        size = sum([dfa_concept.size for dfa_concept in dfa_concepts]) - (N - 1)*(2*b_E + 1) - (N - 1)*(2*b_Q1)

        return DFAProductConcept(dfa_concepts, size)

    def __contains__(self, path: Path) -> bool:
        return all((path in c) for c in self.dfa_concepts)

if __name__=="__main__":

    reference = None

    accepting = ['a', 'abaa', 'bb']
    rejecting = ['abb', 'b']
    data = LabeledExamples(positive=accepting, negative=rejecting)

    for iter in range(5):
        concept = DFAProductConcept.from_examples(
            data=data,
            filter_pred=None,
            alphabet=None,
            order_by_stutter=False,
            temp=1,
            ref=reference
        ) 

        for _ in range(10):
            print(concept)
        input()

