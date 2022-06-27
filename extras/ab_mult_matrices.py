import numpy as np

A = np.array([[3,  4], 
              [1,  2]]) 
B = np.array([[6,  2], 
              [3,  2]])
            
AB = A.dot(B)
BA = B.dot(A)

print(AB)
print(BA)


a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
ab = a.dot(b)
print(ab)


