import statistics

#529.75
# initializing list
test_list = [88, 79.55, 84, 86.88, 84.88, 86.66]
print('sum:', sum(test_list))
average = sum(test_list)/len(test_list)
res = statistics.pstdev(test_list)

print(average, res)
