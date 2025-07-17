def generate_combinations(templist, possibilities, current_combination=[], length=2):
    if len(current_combination) == length:
        possibilities.add(tuple(current_combination))
        return

    for i in range(len(templist)):
        generate_combinations(templist[:i] + templist[i+1:], possibilities, current_combination + [templist[i]], length)

list_data = "a,a,b,b,c,c,c,d".split(",")
templist = list_data
possibilities = set()

selection_length = 3

generate_combinations(templist, possibilities, length=selection_length)

sorted_possibilities = sorted(list(possibilities))

for combo in sorted(sorted_possibilities):
    print(combo)
print(len(possibilities))
