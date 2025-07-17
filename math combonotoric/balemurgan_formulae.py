import math

list = "a,b,b,c,c,c,c".split(",")
choose = 3

list_count_dict = {}
for letter in list:
    if letter in list_count_dict:
        list_count_dict[letter] += 1
    else:
        list_count_dict[letter] = 1

print(list_count_dict)

total_possib = choose ** len(set(list))

print(total_possib)
for val, num in list_count_dict.items():
    if num < choose:
        for i in range(choose - num, choose):
            #print(choose - i)
            #print(i)
            #print(math.comb(len(list_count_dict) - 1, choose - i))
            print(math.factorial(choose) / math.factorial(i))
            #print(choose - i)
