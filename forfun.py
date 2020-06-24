import statistics

#529.75
# initializing list
test_list = [80.44, 88.44, 85.33, 82, 86.88, 85.77]
print('sum:', sum(test_list))
average = sum(test_list)/len(test_list)
res = statistics.pstdev(test_list)

print(average, res)
