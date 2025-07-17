from itertools import groupby

N = int(input())

DAYS = []

for i in range(N):
    DAYS.append(input())

possible_maximums = []

ultimax = []
for i in range(len(DAYS)):

    if "S" not in DAYS[i]:
        temp = DAYS.copy()
        temp[i] = "S"
        res = [list(y) for x, y in groupby(temp)]
        print(res)
        max = 0
        for i in res:
            if "S" in i and len(i) > max:
                max = len(i)
        ultimax.append(max)

print(ultimax)

print(ultimax.max)