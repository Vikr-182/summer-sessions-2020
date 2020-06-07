from numpy import array
from scipy.linalg import svd
A = array([[2,2,2,2],[1.7,0.1,-1.7,-0.1],[0.6,1.8,-0.6,-1.8]])
U,S,V = svd(A)
print(U)
print(S)
print(V)
