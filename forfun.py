import statistics

#529.75
# initializing list
test_list = [64.22, 68.22, 69.56, 61.33, 70.67]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
