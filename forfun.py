import statistics

#529.75
# initializing list
test_list = [88, 94, 89.77, 87.55, 87.55, 92.88]
print('sum:', sum(test_list))
average = sum(test_list)/len(test_list)
res = statistics.pstdev(test_list)

print(average, res)
