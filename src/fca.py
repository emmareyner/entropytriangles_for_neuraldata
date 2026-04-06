"""Formal Concept Analysis helpers.

This module implements Next Closure and a small helper to concatenate extents
and intents across several binary contexts.
"""

from __future__ import annotations

import numpy as np


def compute_concepts(context_bin, object_names=None, attribute_names=None):
    context_bin = (context_bin > 0).astype(int)
    n_objects, n_attributes = context_bin.shape
    if object_names is None:
        object_names = [f'o{i}' for i in range(n_objects)]
    if attribute_names is None:
        attribute_names = [f'a{i}' for i in range(n_attributes)]

    def extent_of(intent):
        if len(intent) == 0:
            return set(range(n_objects))
        mask = np.all(context_bin[:, list(intent)] == 1, axis=1)
        return set(np.where(mask)[0])

    def intent_of(extent):
        if len(extent) == 0:
            return set(range(n_attributes))
        mask = np.all(context_bin[list(extent), :] == 1, axis=0)
        return set(np.where(mask)[0])
    concepts = []
    A = set()
    while True:
        extent = extent_of(A)
        intent = intent_of(extent)
        concepts.append((extent, intent))
        found = False
        for i in reversed(range(n_attributes)):
            if i not in A:
                A_candidate = A | {i}
                closure = intent_of(extent_of(A_candidate))
                if all(((j in A) == (j in closure) for j in range(i))):
                    A = closure
                    found = True
                    break
        if not found:
            break
    n_concepts = len(concepts)
    extents_matrix = np.zeros((n_concepts, n_objects), dtype=int)
    intents_matrix = np.zeros((n_concepts, n_attributes), dtype=int)
    for idx, (extent, intent) in enumerate(concepts):
        extents_matrix[idx, list(extent)] = 1
        intents_matrix[idx, list(intent)] = 1
    return (concepts, extents_matrix, intents_matrix, object_names, attribute_names)


def build_extents_intents_all_worms(contexts_bin, odor_names=None, neuron_names=None):
    n_worms, n_odors, n_neurons = contexts_bin.shape
    if odor_names is None:
        odor_names = [f'o{i}' for i in range(n_odors)]
    if neuron_names is None:
        neuron_names = [f'a{i}' for i in range(n_neurons)]
    all_extents = []
    all_intents = []
    for w in range(n_worms):
        Cw = contexts_bin[w]
        concepts, extents_matrix, intents_matrix, _, _ = compute_concepts(Cw, object_names=odor_names, attribute_names=neuron_names)
        all_extents.append(extents_matrix)
        all_intents.append(intents_matrix)
    ext_all = np.vstack(all_extents)
    int_all = np.vstack(all_intents)
    return (ext_all, int_all)
