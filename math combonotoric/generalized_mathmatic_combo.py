from math import factorial
from itertools import product

def solve_multiset_permutations(multiset, l):
    counts = {}
    for num in multiset:
        counts[num] = counts.get(num, 0) + 1

    distinct_elements = list(counts.keys())
    multiplicities = list(counts.values())
    n = len(distinct_elements)

    possible_values = [range(min(m + 1, l + 1)) for m in multiplicities]

    valid_combinations = [comb for comb in product(*possible_values) if sum(comb) == l]

    total_permutations = 0
    for comb in valid_combinations:
        numerator = factorial(l)
        denominator = 1
        for x in comb:
            denominator *= factorial(x)
        total_permutations += numerator // denominator

    return total_permutations

multiset = [2,2,3,3,3,4,5]
l = 4

result = solve_multiset_permutations(multiset, l)
print(f"Number of unique {l}-digit numbers: {result}")
