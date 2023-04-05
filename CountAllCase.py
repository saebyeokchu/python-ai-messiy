import sys
import pandas as pd
import openpyxl
import numpy as np
from numpy import array

# 1. 엑셀로 불러오기

# 2. 빈도 수 기반으로 카운팅하기, 토요일부터 시작
processed_excel = [
    # 토, 일, 월, 화, 수, 목, 
    ['jisoo','lisa','rose','jeny','lisa','rose','jeny'],
    ['jisoo','lisa','rose','jeny','lisa','rose','jeny'],
    ['jisoo','lisa','rose','jeny','lisa','rose','jeny'],
    ['jisoo','lisa','rose','jeny','lisa','rose','jeny']
]

operators = np.array(processed_excel).flatten()


counter = {}
for operator in operators :
    if operator in counter.keys() :
        counter[operator] += 1
    else : 
        counter[operator] = 1
    
operators_sorted = sorted(counter.items(), key=lambda x:x[1], reverse = True)

print("=" * 23)
print(operators_sorted)
print("=" * 23)

#많이 나온 순서대로 인덱스 부여
i = 0
operators_to_index = {}
for (operator, frequency) in operators_sorted :
    if frequency > 1 :
        i += 1
        operators_to_index[operator] = i
        operators[operators==operator] = i

print(operators_to_index)
        

#2~5개 조각으로 검사
last_pattern = []
last_dict = {}
last_array = []
for j in range(2,6) :
    patterns = []
    for i in range(len(operators)) : 
        if i == len(operators) - j - 1 :
            break
        pattern = operators[i:i+j]
        
        patterns.append(pattern)
    
    unique, counts = np.unique(patterns,axis = 0, return_counts=True)
    temp = []
    k = 0
    for u in unique :
        temp.append(np.array2string(u))
        last_array.append([np.array2string(u), counts[k]])
        k += 1
        
    
    unique_dict = dict(zip(temp, counts))
    last_dict.update(unique_dict)  
    unique_sorted = sorted(unique_dict.items(), key=lambda x:x[1], reverse=True)
    
    last_pattern.append(unique_sorted)

df = pd.DataFrame(last_array, columns=['pattern','count'])
df = df.sort_values('count', ascending=False)

print(df.head(5))
