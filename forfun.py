import statistics

#529.75
# initializing list
test_list = [79.77, 80.44, 75.33, 77.55, 83.33, 75.11]
print('sum:', sum(test_list))
average = sum(test_list)/len(test_list)
res = statistics.pstdev(test_list)

print(average, res)
