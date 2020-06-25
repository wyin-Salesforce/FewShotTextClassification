import statistics

#529.75
# initializing list
test_list = [70.22, 68.44, 76.44, 76.44, 75.78, 77.56]
print('sum:', sum(test_list))
average = sum(test_list)/len(test_list)
res = statistics.pstdev(test_list)

print(average, res)
