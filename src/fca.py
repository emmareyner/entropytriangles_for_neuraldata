"""
fca.py

Formal Concept Analysis helpers: a correct Next Closure (Ganter, 1984) and a helper
that stacks the extents and intents of several binary contexts (e.g. virtual worms).

Public API:
    compute_concepts(context_bin, object_names=None, attribute_names=None)
        -> concepts, extents_matrix, intents_matrix, object_names, attribute_names
    build_extents_intents_all_worms(contexts_bin, odor_names=None, neuron_names=None)
        -> ext_all, int_all

Note on earlier versions
------------------------
Versions of this file before November 2026 built the Next Closure candidate as
A ∪ {i}. The algorithm requires A ⊕ i = ((A ∩ {0, …, i-1}) ∪ {i})'' : the attributes
of A larger than i must be dropped before closing. Keeping them made the enumeration
skip intents, so only a subset of the formal concepts was returned (e.g. [[0,1],[1,0]]
has 4 concepts and the old code returned 3). Any result computed with those versions
should be recomputed.

Implementation notes
--------------------
Sets of objects/attributes are encoded as Python integers (bitmasks). Concepts are
returned in lectic order of their intents. The order of the rows does not affect any
entropy, which only depends on the multiset of extent/intent rows.

Run  to execute the self-test (comparison against brute force).
"""

import numpy as np


# ============================================================
# 1. Formal Concept Analysis (correct Next Closure)
# ============================================================

def compute_concepts(context_bin, object_names=None, attribute_names=None):
    """
    Computes ALL formal concepts of a binary context with Next Closure.

    Parameters
    ----------
    context_bin : array-like (n_objects × n_attributes)
        Binary 0/1 matrix; entry (g, m) = 1 if object g has attribute m.
    object_names, attribute_names : list[str], optional
        Names of objects and attributes (default o0, o1, ... / a0, a1, ...).

    Returns
    -------
    concepts : list[tuple[set[int], set[int]]]
        List of (extent, intent) pairs.
    extents_matrix : np.ndarray (n_concepts × n_objects), 0/1
    intents_matrix : np.ndarray (n_concepts × n_attributes), 0/1
    object_names, attribute_names : list[str]
    """
    context_bin = (np.asarray(context_bin) > 0).astype(int)
    if context_bin.ndim != 2:
        raise ValueError("context_bin must be a 2-D matrix.")
    n_objects, n_attributes = context_bin.shape

    if object_names is None:
        object_names = [f"o{i}" for i in range(n_objects)]
    if attribute_names is None:
        attribute_names = [f"a{i}" for i in range(n_attributes)]

    all_objects = (1 << n_objects) - 1
    all_attributes = (1 << n_attributes) - 1

    # Column m as a bitmask over objects: objects that have attribute m.
    col_masks = []
    for m in range(n_attributes):
        mask = 0
        for g in np.flatnonzero(context_bin[:, m]):
            mask |= 1 << int(g)
        col_masks.append(mask)

    def extent_of(intent_mask):
        """B' : objects having every attribute in B."""
        ext = all_objects
        m = 0
        b = intent_mask
        while b:
            if b & 1:
                ext &= col_masks[m]
            b >>= 1
            m += 1
        return ext

    def intent_of(extent_mask):
        """A' : attributes shared by every object in A (all attributes if A = ∅)."""
        it = 0
        for m in range(n_attributes):
            if col_masks[m] & extent_mask == extent_mask:
                it |= 1 << m
        return it

    def closure(intent_mask):
        return intent_of(extent_of(intent_mask))

    # ------------------------------------------------------------
    # Next Closure: enumerate closed intents in lectic order.
    # Attribute i is "smaller" than j if i < j; prefix(A, i) = A ∩ {0..i-1}.
    # ------------------------------------------------------------
    concepts_masks = []
    A = closure(0)
    concepts_masks.append((extent_of(A), A))

    while A != all_attributes:
        found = False
        for i in reversed(range(n_attributes)):
            bit = 1 << i
            if A & bit:
                continue
            prefix = A & (bit - 1)                 # A ∩ {0, …, i-1}
            B = closure(prefix | bit)              # (A ∩ {<i}) ∪ {i}, closed
            if B & (bit - 1) == prefix:            # lectic condition
                A = B
                concepts_masks.append((extent_of(A), A))
                found = True
                break
        if not found:                              # cannot happen unless A is the last intent
            break

    # ------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------
    n_concepts = len(concepts_masks)
    extents_matrix = np.zeros((n_concepts, n_objects), dtype=int)
    intents_matrix = np.zeros((n_concepts, n_attributes), dtype=int)
    concepts = []

    for idx, (ext_mask, int_mask) in enumerate(concepts_masks):
        ext = {g for g in range(n_objects) if ext_mask >> g & 1}
        it = {m for m in range(n_attributes) if int_mask >> m & 1}
        concepts.append((ext, it))
        if ext:
            extents_matrix[idx, sorted(ext)] = 1
        if it:
            intents_matrix[idx, sorted(it)] = 1

    return concepts, extents_matrix, intents_matrix, object_names, attribute_names


# ============================================================
# 2. FCA over several contexts (e.g. virtual worms)
# ============================================================

def build_extents_intents_all_worms(contexts_bin, odor_names=None, neuron_names=None):
    """
    Applies FCA to every context and stacks all extents and intents.

    Parameters
    ----------
    contexts_bin : np.ndarray (n_worms × n_odors × n_neurons), 0/1

    Returns
    -------
    ext_all : np.ndarray (total_concepts × n_odors)
    int_all : np.ndarray (total_concepts × n_neurons)
    """
    contexts_bin = np.asarray(contexts_bin)
    n_worms, n_odors, n_neurons = contexts_bin.shape

    if odor_names is None:
        odor_names = [f"o{i}" for i in range(n_odors)]
    if neuron_names is None:
        neuron_names = [f"a{i}" for i in range(n_neurons)]

    all_extents, all_intents = [], []
    for w in range(n_worms):
        _, extents_matrix, intents_matrix, _, _ = compute_concepts(
            contexts_bin[w],
            object_names=odor_names,
            attribute_names=neuron_names,
        )
        all_extents.append(extents_matrix)
        all_intents.append(intents_matrix)

    return np.vstack(all_extents), np.vstack(all_intents)


# ============================================================
# 3. Self-test against brute force
# ============================================================

def _brute_force_intents(context_bin):
    """All closed intents by closing every subset of attributes (small contexts only)."""
    from itertools import combinations
    C = (np.asarray(context_bin) > 0).astype(int)
    n, m = C.shape
    intents = set()
    for r in range(m + 1):
        for B in combinations(range(m), r):
            ext = np.all(C[:, list(B)] == 1, axis=1) if B else np.ones(n, bool)
            it = np.all(C[ext] == 1, axis=0) if ext.any() else np.ones(m, bool)
            intents.add(tuple(int(j) for j in np.flatnonzero(it)))
    return intents


if __name__ == "__main__":
    # Smallest case missed by the pre-2026 version
    C = np.array([[0, 1], [1, 0]])
    n = len(compute_concepts(C)[0])
    assert n == 4, n
    print("[[0,1],[1,0]] -> 4 concepts: OK")

    rng = np.random.default_rng(0)
    n_tests = 0
    for _ in range(500):
        n_obj, n_att = rng.integers(1, 15), rng.integers(1, 9)
        p = rng.random()
        C = (rng.random((n_obj, n_att)) < p).astype(int)
        concepts, E, I, _, _ = compute_concepts(C)
        got = [tuple(sorted(it)) for _, it in concepts]
        assert len(got) == len(set(got)), "duplicated concept"
        assert set(got) == _brute_force_intents(C), "missing or spurious concept"
        # each row is a formal concept: A' = B and B' = A
        for k in range(E.shape[0]):
            ext = E[k].astype(bool)
            itm = I[k].astype(bool)
            assert np.array_equal(np.all(C[:, itm] == 1, axis=1) if itm.any() else np.ones(C.shape[0], bool), ext)
        n_tests += 1
    print(f"{n_tests} random contexts: identical to brute force: OK")
