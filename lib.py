from sympy import Rational, symbols, sqrt, pi, zeros
from sympy.physics.wigner import gaunt
from sympy.combinatorics import Permutation
from itertools import combinations, permutations
from collections import Counter

# Core Definitions
f0, f2, f4, f6 = symbols('f0 f2 f4 f6')

orbital_l = {'s': 0, 'p': 1, 'd': 2, 'f': 3}

def get_single_states(l):
    return [(m_l, Rational(1,2)) for m_l in range(-l, l+1)] + [(m_l, Rational(-1,2)) for m_l in range(-l, l+1)]

def enumerate_fn_configurations(n, target_m_l, target_m_s, single_electron_states):
    valid_configs = []
    for config in combinations(single_electron_states, n):
        total_m_l = sum(state[0] for state in config)
        total_m_s = sum(state[1] for state in config)
        if total_m_l == target_m_l and total_m_s == Rational(target_m_s):
            valid_configs.append(config)
    return valid_configs

def coulomb_contribution(m1, m2, m3, m4, l):
    results = []

    if l == 3:
        pre01 = -1 if m1 in [-3, -1, 1, 3] else 1
        pre02 = -1 if m3 in [-3, -1, 1, 3] else 1
        for k, fk, pre in zip([0, 2, 4, 6], [f0, f2, f4, f6], [1, 15, 33, Rational(429, 5)]):
            T1 = pre01 * pre * sqrt(4*pi) * gaunt(l, k, l, -m1, m1-m4, m4) / sqrt(2*k+1)
            T2 = pre02 * pre * sqrt(4*pi) * gaunt(l, k, l, -m3, m3-m2, m2) / sqrt(2*k+1)
            results.append(fk * T1 * T2)

    elif l == 2:
        pre01 = -1 if m1 in [-1, 1] else 1
        pre02 = -1 if m3 in [-1, 1] else 1
        for k, fk, pre in zip([0, 2, 4], [f0, f2, f4], [1, 7, 21]):
            T1 = pre01 * pre * sqrt(4*pi) * gaunt(l, k, l, -m1, m1-m4, m4) / sqrt(2*k+1)
            T2 = pre02 * pre * sqrt(4*pi) * gaunt(l, k, l, -m3, m3-m2, m2) / sqrt(2*k+1)
            results.append(fk * T1 * T2)

    elif l == 1:
        pre01 = -1 if m1 in [-1, 1] else 1
        pre02 = -1 if m3 in [-1, 1] else 1
        for k, fk, pre in zip([0, 2], [f0, f2], [1, 5]):
            T1 = pre01 * pre * sqrt(4*pi) * gaunt(l, k, l, -m1, m1-m4, m4) / sqrt(2*k+1)
            T2 = pre02 * pre * sqrt(4*pi) * gaunt(l, k, l, -m3, m3-m2, m2) / sqrt(2*k+1)
            results.append(fk * T1 * T2)

    elif l == 0:
        return 0

    return sum(results)

def get_required_coulomb_terms(state1, state2):
    terms = []
    if state1 == state2:
        for i in range(len(state1)):
            for j in range(i+1, len(state1)):
                (ml1, ms1) = state1[i]
                (ml2, ms2) = state1[j]
                terms.append((+1, "direct", (ml1, ml2, ml1, ml2)))
                if ms1 == ms2:
                    terms.append((+1, "exchange", (ml1, ml2, ml2, ml1)))
        return terms

    c1 = Counter(state1)
    c2 = Counter(state2)
    removed = list((c1 - c2).elements())
    added = list((c2 - c1).elements())

    if len(removed) != 2 or len(added) != 2:
        return []

    for (a, b) in permutations(removed, 2):
        for (c, d) in permutations(added, 2):
            if a[1] == c[1] and b[1] == d[1]:
                trial_state = list(state1)
                try:
                    idx_a = trial_state.index(a)
                    idx_b = trial_state.index(b)
                except ValueError:
                    continue
                trial_state[idx_a] = c
                trial_state[idx_b] = d
                if Counter(trial_state) != Counter(state2):
                    continue
                perm = [trial_state.index(x) for x in state2]
                sign = Permutation(perm).signature()
                ml1, ml2 = a[0], b[0]
                ml3, ml4 = c[0], d[0]
                terms.append((sign, "direct", (ml1, ml2, ml3, ml4)))
                if a[1] == b[1]:
                    terms.append((sign, "exchange", (ml1, ml2, ml4, ml3)))
                return terms
    return []