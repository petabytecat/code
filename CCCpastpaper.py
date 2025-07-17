# 2024 Junior 1

"""R = int(input())
G = int(input())
B = int(input())

print(R * 3 + G * 4 + B * 5)
"""

# 2024 Junior 2

"""D = int(input())

while True:
    Y = int(input())
    if D > Y:
        D += Y
    else:
        print(D)
        break"""

# 2024 Junior 3

"""N = int(input())

ALL = []
for i in range(N):
    ALL.append(int(input()))

bronze_score = sorted(list(set(ALL)), reverse=True)[2]
bronze_participants = ALL.count(bronze_score)
print(str(bronze_score) + " " + str(bronze_participants))"""

# 2024 Junior 4

ORIGINAL = input()
TROUBLE = input()

original_dictionary = {key:ORIGINAL.count(key) for key in set(ORIGINAL + TROUBLE)}
silly_dictionary = {key:TROUBLE.count(key) for key in set(ORIGINAL + TROUBLE)}

possible_quiet = []
quiet = False
silly_original = False
silly_trouble = False

#silly trouble
for key in set(TROUBLE):
    if silly_dictionary[key] > original_dictionary[key]:
        silly_trouble = key
    if silly_trouble != False:
        break

#quiet
for key in set(ORIGINAL):
    if silly_dictionary[key] == 0:
        possible_quiet.append(key)

#silly og
for key in set(ORIGINAL):
    if silly_dictionary[key] < original_dictionary[key] and original_dictionary[silly_trouble] <= silly_dictionary[key] and silly_dictionary[key] != 0:
        silly_original = key
    if silly_original != False:
        break

refined_quiet = possible_quiet.remove(silly_original)

if not refined_quiet:
    quiet = False
else:
    quiet = refined_quiet[0]


if silly_original != False and silly_trouble != False:
    print(silly_original + " " + silly_trouble) 
else:
    print("- -")

if quiet != False:
    print(quiet)
else:
    print("-")

