import statistics

#529.75
# initializing list
test_list = [80.67, 84, 87.11, 87.33, 80.44]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
