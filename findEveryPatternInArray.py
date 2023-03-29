import numpy as np

def findAllPattern(arr) :
  counts = {}
  patterns = [
    arr[i : i+n] for n in range(2,len(arr)) for i in range(len(arr)-n+1)
    if all(x1 == x2 for (x1, x2) in zip(np.array(arr[i:i+n]), np.array(arr[i+n:i+2*n])))
  ]

  for pattern in patterns :
    if len(pattern) < 6 and len(pattern) > 1 :
      if tuple(pattern) in counts and counts[tuple(pattern)] > 0 :
        counts[tuple(pattern)] = counts[tuple(pattern)] + 1
      else :
        counts[tuple(pattern)] = 1
  
  patterns = list(set(tuple(pattern) for pattern in patterns if len(pattern) < 6))

  print("========등장한 모든 패턴==========")
  for key in counts : 
    print(key , " : ",counts[key])

def countEachElement(arr) :
  unique, counts = np.unique(arr, return_counts=True)
  counts = dict(zip(unique, counts))
  
  print("========각 경우의 수 카운트==========")
  for key in counts : 
    print(key , " : ",counts[key])
    
str = '2,3,4,2,3,4'
arr = np.array(str.split(','))

findAllPattern(arr)
countEachElement(arr)
